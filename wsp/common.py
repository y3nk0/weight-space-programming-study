"""Shared utilities: model loading, prompt templates, greedy evaluation,
forward-pass budgeting, and weight-delta save/apply."""
import math, re, time, unicodedata, json, os
from typing import List, Tuple, Sequence, Dict, Optional
import torch

MODEL_ID = os.environ.get("WSP_MODEL", "Qwen/Qwen2.5-0.5B")
TEMPLATE = "Input: {x}\nOutput:"          # zero-shot query (also the ICL query)
MAX_NEW_TOKENS = 6                        # per task: set with set_decoding_length() (Qwen tokenizes digits one by one)


def max_new_tokens_for(items, tok, floor=6, cap=16):
    """Enough decoding steps for the longest gold answer (+2 for a newline / stop)."""
    longest = max(len(tok(" " + ys[0], add_special_tokens=False)["input_ids"]) for _, ys in items)
    return max(floor, min(cap, longest + 2))


def set_decoding_length(items, tok):
    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = max_new_tokens_for(items, tok)
    return MAX_NEW_TOKENS
DEMO_TEMPLATE = "Input: {x}\nOutput: {y}\n\n"

TEMPLATES = {                                   # the evaluation wordings; "train" is the one the editor is normally scored on
    "train": ("Input: {x}\nOutput:", "Input: {x}\nOutput: {y}\n\n"),
    "qa": ("Q: {x}\nA:", "Q: {x}\nA: {y}\n\n"),
    "arrow": ("{x} ->", "{x} -> {y}\n"),
    "word_answer": ("word: {x}\nanswer:", "word: {x}\nanswer: {y}\n\n"),
}


def set_template(name):
    global TEMPLATE, DEMO_TEMPLATE
    TEMPLATE, DEMO_TEMPLATE = TEMPLATES[name]
    return TEMPLATE

# ---------------------------------------------------------------- long-form tasks (final-number scoring)
SCORER = None          # optional task scorer(generation, answers) -> bool; None = exact match on the first line
STOP_STRINGS = None    # optional stop strings for generation
SCORING_TEXT = None    # how the metric is described to the editor
LONG_FORM_TASKS = {"gsm8k": 256}


def final_number(text):
    """The number after 'The answer is', else the last number in the text (cut at the next 'Input:')."""
    if text is None: return None
    for stop in ("\nInput:", "\n\nQuestion:"):
        if stop in text: text = text[:text.index(stop)]
    m = re.search(r"[Tt]he answer is\s*\$?\s*(-?\d[\d,]*\.?\d*)", text)
    if not m:
        ms = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
        if not ms: return None
        val = ms[-1]
    else: val = m.group(1)
    val = val.replace(",", "").rstrip(".")
    try: return str(int(float(val))) if float(val) == int(float(val)) else str(float(val))
    except ValueError: return val


def set_task_mode(task):
    """Switch scoring and decoding for long-form tasks (and back for the word-mapping tasks)."""
    global SCORER, STOP_STRINGS, SCORING_TEXT, MAX_NEW_TOKENS
    if task in LONG_FORM_TASKS:
        MAX_NEW_TOKENS = LONG_FORM_TASKS[task]; STOP_STRINGS = ["\nInput:"]
        SCORER = lambda g, ys: final_number(g) is not None and final_number(g) == final_number(ys[0])
        SCORING_TEXT = ("The generated text (cut at the first \"\\nInput:\") counts as correct if the number after \"The answer is\", or otherwise the last number in it, "
                        "equals the reference final number. The reference outputs in train_examples are worked solutions ending in \"The answer is N.\"; only N is scored, "
                        "so the model may answer directly or reason first, as long as the final number is right.")
    else:
        SCORER = STOP_STRINGS = SCORING_TEXT = None
    return MAX_NEW_TOKENS

torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32          # set via --dtype bf16 for models that do not fit in fp32 (e.g. Qwen2.5-14B on a 70 GB slice)
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load_model_and_tokenizer(model_id: str = MODEL_ID, dtype=None):
    dtype = dtype or DTYPE
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)      # transformers >= 5
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    # the GPUs are shared with other users: pick the visible device with the most free memory, wait if none fits
    global DEVICE
    need = sum(p.numel() * p.element_size() for p in model.parameters()) * 1.15
    for attempt in range(int(os.environ.get("WSP_LOAD_RETRIES", "120"))):
        if torch.cuda.is_available():
            free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
            best_free, best = max(free)
            if best_free > need:
                try:
                    DEVICE = torch.device(f"cuda:{best}"); model = model.to(DEVICE)
                    print(f"[load] on cuda:{best} ({best_free/2**30:.0f} GB free)", flush=True); break
                except (RuntimeError, torch.OutOfMemoryError) as e:
                    if "out of memory" not in str(e).lower(): raise
                    model = model.to("cpu"); torch.cuda.empty_cache()
            print(f"[load] no GPU with {need/2**30:.0f} GB free (best {best_free/2**30:.0f} GB); retrying in 60s (attempt {attempt + 1})", flush=True)
        time.sleep(60)
    else:
        raise RuntimeError("GPU memory unavailable after repeated attempts")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


FIXED_DEMOS: list = []   # hybrid condition: k fixed demonstrations that every "zero-shot" prompt carries


def make_prompt(x: str, demos: Optional[Sequence[Tuple[str, str]]] = None) -> str:
    s = ""
    if demos is None and FIXED_DEMOS:
        demos = FIXED_DEMOS
    if demos:
        s = "".join(DEMO_TEMPLATE.format(x=dx, y=dy) for dx, dy in demos)
    return s + TEMPLATE.format(x=x)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    s = s.split("\n")[0].strip()
    s = re.sub(r"[\.\,\;\:\!\?\"“”‘’]+$", "", s).strip()
    s = s.strip("'\"")
    return s


def is_correct(generation: str, answers: Sequence[str]) -> bool:
    g = normalize(generation)
    if not g:
        return False
    for a in answers:
        a = normalize(a)
        if g == a or g.startswith(a + " ") or g.startswith(a + ","):
            return True
    return False


@torch.no_grad()
def greedy_generate(model, tok, prompts: List[str], max_new_tokens: int = 6,
                    batch_size: int = 32) -> List[str]:
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        extra = {"stop_strings": STOP_STRINGS, "tokenizer": tok} if STOP_STRINGS else {}
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, **extra)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        if STOP_STRINGS:
            texts = [min((t[:t.index(st)] for st in STOP_STRINGS if st in t), key=len, default=t) for t in texts]
        outs.extend(texts)
    return outs


@torch.no_grad()
def evaluate(model, tok, items: Sequence[Tuple[str, Sequence[str]]],
             demos: Optional[Sequence[Tuple[str, str]]] = None,
             max_new_tokens: Optional[int] = None, batch_size: int = 32) -> Dict:
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
    prompts = [make_prompt(x, demos) for x, _ in items]
    gens = greedy_generate(model, tok, prompts, max_new_tokens, batch_size)
    correct = [(SCORER(g, ys) if SCORER else is_correct(g, ys)) for g, (_, ys) in zip(gens, items)]
    return {"accuracy": sum(correct) / max(1, len(items)),
            "n": len(items),
            "generations": [(final_number(g) or "") if SCORER else normalize(g) for g in gens],
            "raw_generations": gens,
            "correct": correct}


@torch.no_grad()
def last_token_logits(model, tok, prompts: List[str], batch_size: int = 32) -> torch.Tensor:
    """Logits at the final position (left padding => position -1)."""
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        try:
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]      # avoid a [batch, seq, 152K] logits tensor
        except TypeError:
            logits = model(**enc).logits[:, -1, :]
        out.append(logits.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def last_token_hidden(model, tok, prompts: List[str], batch_size: int = 16) -> torch.Tensor:
    """Residual stream at the final position for every layer: [n_prompts, n_layers+1, d]."""
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        try:
            hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
        except TypeError:
            hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1, :].float().cpu() for h in hs], dim=1))
    return torch.cat(out)


# ---------------------------------------------------------------- budgeting
class BudgetExceeded(RuntimeError):
    pass


class ForwardBudget:
    """Counts model forward calls.  One 'pass' = one forward call with <= batch_cap
    sequences; larger batches cost ceil(batch/batch_cap) passes.  Each decoding step
    inside generate() is one forward call."""

    def __init__(self, model, max_passes: int, batch_cap: int = 64, generate_as_one: bool = False):
        self.model, self.max_passes, self.batch_cap = model, max_passes, batch_cap
        self.used = 0
        self.log = []
        self._orig_forward = model.forward
        model.forward = self._forward
        self._in_generate = False; self.generate_as_one = generate_as_one
        if generate_as_one:                       # long-form variant: one generate() call = one pass per <= batch_cap prompts
            self._orig_generate = model.generate
            model.generate = self._generate

    def _generate(self, *args, **kwargs):
        ids = kwargs.get("input_ids", None)
        if ids is None and args: ids = args[0]
        bs = int(ids.shape[0]) if ids is not None else 1
        cost = max(1, math.ceil(bs / self.batch_cap))
        if self.used + cost > self.max_passes:
            raise BudgetExceeded(f"Forward-pass budget exhausted: used {self.used}/{self.max_passes}; this generate() call would cost {cost} (batch={bs}).")
        self.used += cost; self.log.append({"batch": bs, "seq": int(ids.shape[1]) if ids is not None else 1, "cost": cost, "generate": True, "t": time.time()})
        self._in_generate = True
        try: return self._generate_split(*args, **kwargs)
        finally: self._in_generate = False

    def _generate_split(self, *args, **kwargs):
        """Run generate; on GPU out-of-memory (shared GPU), split the batch in two and pad the outputs to a common length."""
        try:
            return self._orig_generate(*args, **kwargs)
        except torch.OutOfMemoryError:
            ids = kwargs.get("input_ids", None); am = kwargs.get("attention_mask", None)
            if ids is None or ids.shape[0] < 2: raise
            torch.cuda.empty_cache(); h = ids.shape[0] // 2
            parts = []
            for sl in (slice(0, h), slice(h, None)):
                kw = dict(kwargs); kw["input_ids"] = ids[sl]
                if am is not None: kw["attention_mask"] = am[sl]
                parts.append(self._generate_split(*args, **kw))
            L = max(p.shape[1] for p in parts); pad = kwargs.get("pad_token_id", 0) or 0
            parts = [torch.nn.functional.pad(p, (0, L - p.shape[1]), value=pad) for p in parts]
            return torch.cat(parts, 0)

    def _forward(self, *args, **kwargs):
        if self._in_generate:
            return self._orig_forward(*args, **kwargs)
        ids = kwargs.get("input_ids", None)
        if ids is None and args:
            ids = args[0]
        if ids is None and kwargs.get("inputs_embeds") is not None:
            ids = kwargs["inputs_embeds"]
        bs = int(ids.shape[0]) if ids is not None else 1
        seq = int(ids.shape[1]) if (ids is not None and ids.dim() > 1) else 1
        cost = max(1, math.ceil(bs / self.batch_cap))
        if self.used + cost > self.max_passes:
            raise BudgetExceeded(
                f"Forward-pass budget exhausted: used {self.used}/{self.max_passes}; "
                f"this call would cost {cost} (batch={bs}).")
        self.used += cost
        self.log.append({"batch": bs, "seq": seq, "cost": cost, "t": time.time()})
        return self._orig_forward(*args, **kwargs)

    @property
    def remaining(self):
        return self.max_passes - self.used

    def detach(self):
        self.model.forward = self._orig_forward
        if self.generate_as_one: self.model.generate = self._orig_generate


# ---------------------------------------------------------------- deltas
def snapshot_params(model, device=None) -> Dict[str, torch.Tensor]:
    """Copy of all parameters (on `device`; defaults to the model's device, CPU for models > 2.5B)."""
    if device is None:
        device = "cpu"      # the GPU is shared; keep the reference copy in host RAM (1 TB available)
    return {n: p.detach().to(device, copy=True) for n, p in model.named_parameters()}


def compute_delta(model, base: Dict[str, torch.Tensor], sparse_threshold: float = 0.05) -> Dict:
    """Return {name: {'dense': tensor} | {'idx': LongTensor, 'val': tensor, 'shape': ...}}
    for every parameter that differs from base."""
    delta, stats = {}, {}
    for n, p in model.named_parameters():
        d = (p.detach().to("cpu", copy=True) - base[n].to("cpu"))     # on the CPU: no GPU allocation on a shared device
        nz = d != 0
        k = int(nz.sum())
        if k == 0:
            continue
        stats[n] = {"changed": k, "numel": d.numel(), "frac": k / d.numel(),
                    "delta_fro": float(d.norm()), "base_fro": float(base[n].norm()),
                    "rel": float(d.norm() / (base[n].norm() + 1e-12))}
        if k / d.numel() < sparse_threshold:
            idx = nz.flatten().nonzero().squeeze(1)
            delta[n] = {"idx": idx.cpu(), "val": d.flatten()[idx].clone().cpu(), "shape": tuple(d.shape)}
        else:
            delta[n] = {"dense": d.clone().cpu()}
    return {"delta": delta, "stats": stats}


def apply_delta(model, delta: Dict[str, Dict]):
    params = dict(model.named_parameters())
    for n, d in delta.items():
        p = params[n]
        if "dense" in d:
            p.data.add_(d["dense"].to(p.device, p.dtype))
        else:
            flat = p.data.view(-1)
            flat[d["idx"].to(p.device)] += d["val"].to(p.device, p.dtype)


def delta_summary(stats: Dict) -> str:
    total_changed = sum(s["changed"] for s in stats.values())
    lines = [f"{len(stats)} tensors changed, {total_changed:,} scalar weights changed"]
    for n, s in sorted(stats.items(), key=lambda kv: -kv[1]["delta_fro"]):
        lines.append(f"  {n}: {s['changed']:,}/{s['numel']:,} ({100*s['frac']:.3f}%) "
                     f"|d|={s['delta_fro']:.4f} rel={s['rel']:.4f}")
    return "\n".join(lines)
