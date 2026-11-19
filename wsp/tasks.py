"""Word-mapping tasks in the style of the ICL "task vector" / "function vector"
literature (Hendel et al. 2023; Todd et al. 2024).  Each item is
(input, [acceptable outputs]).  The first acceptable output is the canonical one
used in demonstrations."""

ANTONYM = [
 ("hot", ["cold"]), ("big", ["small", "little"]), ("fast", ["slow"]), ("up", ["down"]),
 ("happy", ["sad", "unhappy"]), ("tall", ["short"]), ("rich", ["poor"]), ("early", ["late"]),
 ("open", ["closed", "close", "shut"]), ("strong", ["weak"]), ("wet", ["dry"]), ("full", ["empty"]),
 ("hard", ["soft", "easy"]), ("loud", ["quiet", "soft"]), ("clean", ["dirty"]), ("thick", ["thin"]),
 ("wide", ["narrow"]), ("high", ["low"]), ("long", ["short"]), ("heavy", ["light"]),
 ("true", ["false"]), ("good", ["bad", "evil"]), ("black", ["white"]), ("day", ["night"]),
 ("love", ["hate"]), ("win", ["lose"]), ("buy", ["sell"]), ("push", ["pull"]),
 ("give", ["take", "receive"]), ("come", ["go"]), ("start", ["stop", "finish", "end"]),
 ("begin", ["end", "finish"]), ("inside", ["outside"]), ("above", ["below"]), ("before", ["after"]),
 ("always", ["never"]), ("more", ["less"]), ("first", ["last"]), ("top", ["bottom"]),
 ("north", ["south"]), ("east", ["west"]), ("male", ["female"]), ("man", ["woman"]),
 ("boy", ["girl"]), ("king", ["queen"]), ("husband", ["wife"]), ("brother", ["sister"]),
 ("father", ["mother"]), ("son", ["daughter"]), ("friend", ["enemy"]), ("question", ["answer"]),
 ("teacher", ["student"]), ("war", ["peace"]), ("life", ["death"]), ("alive", ["dead"]),
 ("asleep", ["awake"]), ("sweet", ["sour", "bitter"]), ("cheap", ["expensive"]), ("near", ["far"]),
 ("deep", ["shallow"]), ("smooth", ["rough"]), ("sharp", ["dull", "blunt"]),
 ("brave", ["cowardly", "afraid", "fearful"]), ("polite", ["rude", "impolite"]),
 ("safe", ["dangerous", "unsafe"]), ("simple", ["complex", "complicated", "difficult"]),
 ("visible", ["invisible"]), ("possible", ["impossible"]), ("legal", ["illegal"]),
 ("honest", ["dishonest"]), ("patient", ["impatient"]), ("correct", ["incorrect", "wrong"]),
 ("increase", ["decrease"]), ("accept", ["reject", "refuse", "decline"]),
 ("arrive", ["depart", "leave"]), ("remember", ["forget"]), ("laugh", ["cry"]),
 ("attack", ["defend", "defense"]), ("success", ["failure"]), ("victory", ["defeat"]),
 ("maximum", ["minimum"]), ("majority", ["minority"]), ("entrance", ["exit"]),
 ("import", ["export"]), ("include", ["exclude"]), ("ancient", ["modern"]), ("future", ["past"]),
 ("summer", ["winter"]), ("floor", ["ceiling"]), ("tight", ["loose"]), ("fat", ["thin", "skinny"]),
 ("young", ["old"]), ("new", ["old"]), ("sick", ["healthy", "well"]), ("hungry", ["full"]),
 ("noisy", ["quiet", "silent"]), ("bright", ["dark", "dim"]), ("warm", ["cool", "cold"]),
 ("generous", ["stingy", "selfish", "greedy"]), ("optimistic", ["pessimistic"]),
 ("positive", ["negative"]), ("public", ["private"]), ("urban", ["rural"]),
 ("vertical", ["horizontal"]), ("interior", ["exterior"]), ("internal", ["external"]),
 ("input", ["output"]), ("plus", ["minus"]), ("add", ["subtract"]), ("multiply", ["divide"]),
 ("ally", ["enemy"]), ("hero", ["villain"]), ("angel", ["devil", "demon"]), ("heaven", ["hell"]),
 ("dawn", ["dusk"]), ("sunrise", ["sunset"]), ("arrival", ["departure"]), ("birth", ["death"]),
 ("lend", ["borrow"]), ("teach", ["learn"]), ("speak", ["listen"]), ("lock", ["unlock"]),
 ("tie", ["untie"]), ("dress", ["undress"]), ("pass", ["fail"]), ("guilty", ["innocent"]),
 ("permanent", ["temporary"]), ("rare", ["common"]), ("absent", ["present"]),
 ("empty", ["full"]), ("dry", ["wet"]), ("cold", ["hot"]), ("slow", ["fast"]), ("short", ["tall", "long"]),
 ("poor", ["rich"]), ("late", ["early"]), ("weak", ["strong"]), ("thin", ["thick", "fat"]),
 ("low", ["high"]), ("light", ["dark", "heavy"]), ("false", ["true"]), ("bad", ["good"]),
 ("white", ["black"]), ("night", ["day"]), ("hate", ["love"]), ("lose", ["win"]), ("sell", ["buy"]),
 ("pull", ["push"]), ("go", ["come", "stop"]), ("stop", ["go", "start"]), ("end", ["begin", "start"]),
 ("outside", ["inside"]), ("below", ["above"]), ("after", ["before"]), ("never", ["always"]),
 ("less", ["more"]), ("last", ["first"]), ("bottom", ["top"]), ("south", ["north"]), ("west", ["east"]),
 ("female", ["male"]), ("woman", ["man"]), ("girl", ["boy"]), ("queen", ["king"]), ("wife", ["husband"]),
 ("sister", ["brother"]), ("mother", ["father"]), ("daughter", ["son"]), ("enemy", ["friend", "ally"]),
 ("answer", ["question"]), ("student", ["teacher"]), ("peace", ["war"]), ("death", ["life", "birth"]),
 ("dead", ["alive"]), ("awake", ["asleep"]), ("expensive", ["cheap"]), ("far", ["near", "close"]),
 ("shallow", ["deep"]), ("rough", ["smooth"]), ("dull", ["sharp", "bright"]), ("rude", ["polite"]),
 ("dangerous", ["safe"]), ("complex", ["simple"]), ("invisible", ["visible"]),
 ("impossible", ["possible"]), ("illegal", ["legal"]), ("dishonest", ["honest"]),
 ("wrong", ["right", "correct"]), ("decrease", ["increase"]), ("reject", ["accept"]),
 ("depart", ["arrive"]), ("forget", ["remember"]), ("cry", ["laugh"]), ("defend", ["attack"]),
 ("failure", ["success"]), ("defeat", ["victory"]), ("minimum", ["maximum"]),
 ("minority", ["majority"]), ("exit", ["entrance", "enter"]), ("export", ["import"]),
 ("exclude", ["include"]), ("modern", ["ancient", "old"]), ("past", ["future", "present"]),
 ("winter", ["summer"]), ("ceiling", ["floor"]), ("loose", ["tight"]), ("healthy", ["sick", "unhealthy"]),
 ("quiet", ["loud", "noisy"]), ("dark", ["light", "bright"]), ("cool", ["warm", "hot"]),
 ("pessimistic", ["optimistic"]), ("negative", ["positive"]), ("private", ["public"]),
 ("rural", ["urban"]), ("horizontal", ["vertical"]), ("exterior", ["interior"]),
 ("external", ["internal"]), ("output", ["input"]), ("minus", ["plus"]), ("subtract", ["add"]),
 ("divide", ["multiply"]), ("villain", ["hero"]), ("devil", ["angel"]), ("hell", ["heaven"]),
 ("dusk", ["dawn"]), ("sunset", ["sunrise"]), ("departure", ["arrival"]), ("borrow", ["lend"]),
 ("learn", ["teach"]), ("listen", ["speak", "talk"]), ("unlock", ["lock"]), ("fail", ["pass", "succeed"]),
 ("innocent", ["guilty"]), ("temporary", ["permanent"]), ("common", ["rare", "uncommon"]),
 ("present", ["absent", "past"]),
]

CAPITAL = [
 ("France", ["Paris"]), ("Germany", ["Berlin"]), ("Italy", ["Rome"]), ("Spain", ["Madrid"]),
 ("Portugal", ["Lisbon"]), ("Greece", ["Athens"]), ("Japan", ["Tokyo"]), ("China", ["Beijing"]),
 ("Russia", ["Moscow"]), ("Egypt", ["Cairo"]), ("Canada", ["Ottawa"]), ("Australia", ["Canberra"]),
 ("Brazil", ["Brasilia"]), ("Argentina", ["Buenos Aires"]), ("Chile", ["Santiago"]), ("Peru", ["Lima"]),
 ("Colombia", ["Bogota"]), ("Mexico", ["Mexico City"]), ("Cuba", ["Havana"]), ("India", ["New Delhi", "Delhi"]),
 ("Pakistan", ["Islamabad"]), ("Bangladesh", ["Dhaka"]), ("Thailand", ["Bangkok"]), ("Vietnam", ["Hanoi"]),
 ("Indonesia", ["Jakarta"]), ("Philippines", ["Manila"]), ("Malaysia", ["Kuala Lumpur"]),
 ("South Korea", ["Seoul"]), ("North Korea", ["Pyongyang"]), ("Mongolia", ["Ulaanbaatar", "Ulan Bator"]),
 ("Nepal", ["Kathmandu"]), ("Iran", ["Tehran"]), ("Iraq", ["Baghdad"]), ("Syria", ["Damascus"]),
 ("Lebanon", ["Beirut"]), ("Jordan", ["Amman"]), ("Israel", ["Jerusalem"]), ("Saudi Arabia", ["Riyadh"]),
 ("Turkey", ["Ankara"]), ("Afghanistan", ["Kabul"]), ("Kenya", ["Nairobi"]), ("Ethiopia", ["Addis Ababa"]),
 ("Nigeria", ["Abuja"]), ("Ghana", ["Accra"]), ("Senegal", ["Dakar"]), ("Morocco", ["Rabat"]),
 ("Algeria", ["Algiers"]), ("Tunisia", ["Tunis"]), ("Libya", ["Tripoli"]), ("Sudan", ["Khartoum"]),
 ("South Africa", ["Pretoria", "Cape Town"]), ("Zimbabwe", ["Harare"]), ("Zambia", ["Lusaka"]),
 ("Tanzania", ["Dodoma", "Dar es Salaam"]), ("Uganda", ["Kampala"]), ("Angola", ["Luanda"]),
 ("Mozambique", ["Maputo"]), ("Madagascar", ["Antananarivo"]), ("United Kingdom", ["London"]),
 ("Ireland", ["Dublin"]), ("Netherlands", ["Amsterdam"]), ("Belgium", ["Brussels"]),
 ("Switzerland", ["Bern", "Berne"]), ("Austria", ["Vienna"]), ("Poland", ["Warsaw"]),
 ("Czech Republic", ["Prague"]), ("Hungary", ["Budapest"]), ("Romania", ["Bucharest"]),
 ("Bulgaria", ["Sofia"]), ("Serbia", ["Belgrade"]), ("Croatia", ["Zagreb"]), ("Slovenia", ["Ljubljana"]),
 ("Slovakia", ["Bratislava"]), ("Ukraine", ["Kyiv", "Kiev"]), ("Belarus", ["Minsk"]),
 ("Lithuania", ["Vilnius"]), ("Latvia", ["Riga"]), ("Estonia", ["Tallinn"]), ("Finland", ["Helsinki"]),
 ("Sweden", ["Stockholm"]), ("Norway", ["Oslo"]), ("Denmark", ["Copenhagen"]), ("Iceland", ["Reykjavik"]),
 ("Albania", ["Tirana"]), ("North Macedonia", ["Skopje"]), ("Bosnia and Herzegovina", ["Sarajevo"]),
 ("Moldova", ["Chisinau"]), ("Georgia", ["Tbilisi"]), ("Armenia", ["Yerevan"]), ("Azerbaijan", ["Baku"]),
 ("Kazakhstan", ["Astana", "Nur-Sultan"]), ("Uzbekistan", ["Tashkent"]), ("Venezuela", ["Caracas"]),
 ("Ecuador", ["Quito"]), ("Bolivia", ["La Paz", "Sucre"]), ("Uruguay", ["Montevideo"]),
 ("Paraguay", ["Asuncion"]), ("Jamaica", ["Kingston"]), ("Haiti", ["Port-au-Prince"]),
 ("Dominican Republic", ["Santo Domingo"]), ("Guatemala", ["Guatemala City"]),
 ("Honduras", ["Tegucigalpa"]), ("Nicaragua", ["Managua"]), ("Costa Rica", ["San Jose"]),
 ("Panama", ["Panama City"]), ("New Zealand", ["Wellington"]), ("Fiji", ["Suva"]),
 ("Sri Lanka", ["Colombo", "Sri Jayawardenepura Kotte"]), ("Myanmar", ["Naypyidaw", "Yangon"]),
 ("Cambodia", ["Phnom Penh"]), ("Laos", ["Vientiane"]), ("Taiwan", ["Taipei"]), ("Qatar", ["Doha"]),
 ("Kuwait", ["Kuwait City"]), ("Oman", ["Muscat"]), ("Yemen", ["Sanaa", "Sana'a"]),
 ("United Arab Emirates", ["Abu Dhabi"]), ("Cyprus", ["Nicosia"]), ("Malta", ["Valletta"]),
 ("Luxembourg", ["Luxembourg"]), ("Somalia", ["Mogadishu"]), ("Rwanda", ["Kigali"]),
 ("Cameroon", ["Yaounde"]), ("Ivory Coast", ["Yamoussoukro", "Abidjan"]), ("Mali", ["Bamako"]),
 ("Niger", ["Niamey"]), ("Chad", ["N'Djamena", "Ndjamena"]), ("Botswana", ["Gaborone"]),
 ("Namibia", ["Windhoek"]), ("United States", ["Washington", "Washington, D.C.", "Washington D.C."]),
 ("Bahrain", ["Manama"]), ("Bhutan", ["Thimphu"]), ("Brunei", ["Bandar Seri Begawan"]),
 ("Eritrea", ["Asmara"]), ("Gabon", ["Libreville"]), ("Guinea", ["Conakry"]), ("Liberia", ["Monrovia"]),
 ("Malawi", ["Lilongwe"]), ("Mauritania", ["Nouakchott"]), ("Sierra Leone", ["Freetown"]),
 ("Togo", ["Lome"]), ("Burkina Faso", ["Ouagadougou"]), ("Benin", ["Porto-Novo", "Cotonou"]),
 ("Kyrgyzstan", ["Bishkek"]), ("Tajikistan", ["Dushanbe"]), ("Turkmenistan", ["Ashgabat"]),
 ("Papua New Guinea", ["Port Moresby"]), ("Trinidad and Tobago", ["Port of Spain"]),
 ("Bahamas", ["Nassau"]), ("Barbados", ["Bridgetown"]), ("Belize", ["Belmopan"]),
 ("El Salvador", ["San Salvador"]), ("Suriname", ["Paramaribo"]), ("Guyana", ["Georgetown"]),
 ("Montenegro", ["Podgorica"]), ("Liechtenstein", ["Vaduz"]), ("Monaco", ["Monaco"]),
 ("Andorra", ["Andorra la Vella"]), ("San Marino", ["San Marino"]), ("Vatican City", ["Vatican City"]),
]

PAST_TENSE = [
 ("go", ["went"]), ("see", ["saw"]), ("eat", ["ate"]), ("run", ["ran"]), ("take", ["took"]),
 ("give", ["gave"]), ("come", ["came"]), ("write", ["wrote"]), ("speak", ["spoke"]), ("break", ["broke"]),
 ("drive", ["drove"]), ("sing", ["sang"]), ("drink", ["drank"]), ("swim", ["swam"]), ("begin", ["began"]),
 ("know", ["knew"]), ("grow", ["grew"]), ("throw", ["threw"]), ("fly", ["flew"]), ("draw", ["drew"]),
 ("buy", ["bought"]), ("bring", ["brought"]), ("think", ["thought"]), ("teach", ["taught"]),
 ("catch", ["caught"]), ("fight", ["fought"]), ("seek", ["sought"]), ("sell", ["sold"]), ("tell", ["told"]),
 ("find", ["found"]), ("hold", ["held"]), ("stand", ["stood"]), ("understand", ["understood"]),
 ("sit", ["sat"]), ("feel", ["felt"]), ("keep", ["kept"]), ("sleep", ["slept"]), ("leave", ["left"]),
 ("meet", ["met"]), ("lose", ["lost"]), ("send", ["sent"]), ("spend", ["spent"]), ("build", ["built"]),
 ("lend", ["lent"]), ("make", ["made"]), ("say", ["said"]), ("pay", ["paid"]), ("lay", ["laid"]),
 ("have", ["had"]), ("do", ["did"]), ("get", ["got"]), ("forget", ["forgot"]), ("win", ["won"]),
 ("shoot", ["shot"]), ("choose", ["chose"]), ("freeze", ["froze"]), ("steal", ["stole"]), ("wake", ["woke"]),
 ("wear", ["wore"]), ("tear", ["tore"]), ("ride", ["rode"]), ("rise", ["rose"]), ("hide", ["hid"]),
 ("bite", ["bit"]), ("fall", ["fell"]), ("feed", ["fed"]), ("lead", ["led"]), ("hear", ["heard"]),
 ("become", ["became"]), ("forgive", ["forgave"]), ("hang", ["hung"]), ("dig", ["dug"]), ("stick", ["stuck"]),
 ("strike", ["struck"]), ("swing", ["swung"]), ("shake", ["shook"]), ("blow", ["blew"]), ("bear", ["bore"]),
 ("walk", ["walked"]), ("talk", ["talked"]), ("play", ["played"]), ("jump", ["jumped"]), ("look", ["looked"]),
 ("want", ["wanted"]), ("need", ["needed"]), ("help", ["helped"]), ("work", ["worked"]), ("call", ["called"]),
 ("ask", ["asked"]), ("use", ["used"]), ("move", ["moved"]), ("live", ["lived"]), ("love", ["loved"]),
 ("like", ["liked"]), ("start", ["started"]), ("open", ["opened"]), ("close", ["closed"]), ("stop", ["stopped"]),
 ("plan", ["planned"]), ("try", ["tried"]), ("cry", ["cried"]), ("carry", ["carried"]), ("study", ["studied"]),
 ("worry", ["worried"]), ("hurry", ["hurried"]), ("dance", ["danced"]), ("smile", ["smiled"]),
 ("laugh", ["laughed"]), ("watch", ["watched"]), ("wash", ["washed"]), ("push", ["pushed"]),
 ("pull", ["pulled"]), ("climb", ["climbed"]), ("cook", ["cooked"]), ("clean", ["cleaned"]),
 ("paint", ["painted"]), ("visit", ["visited"]), ("enjoy", ["enjoyed"]), ("learn", ["learned", "learnt"]),
 ("dream", ["dreamed", "dreamt"]), ("travel", ["traveled", "travelled"]), ("listen", ["listened"]),
 ("answer", ["answered"]), ("follow", ["followed"]), ("happen", ["happened"]), ("change", ["changed"]),
 ("decide", ["decided"]), ("arrive", ["arrived"]), ("finish", ["finished"]), ("wait", ["waited"]),
 ("kill", ["killed"]), ("fill", ["filled"]), ("kick", ["kicked"]), ("pick", ["picked"]), ("touch", ["touched"]),
 ("agree", ["agreed"]), ("believe", ["believed"]), ("remember", ["remembered"]), ("explain", ["explained"]),
 ("join", ["joined"]), ("wish", ["wished"]), ("hope", ["hoped"]), ("add", ["added"]), ("fix", ["fixed"]),
 ("mix", ["mixed"]), ("drop", ["dropped"]), ("rob", ["robbed"]), ("hug", ["hugged"]), ("beg", ["begged"]),
 ("prefer", ["preferred"]), ("admit", ["admitted"]), ("swear", ["swore"]), ("sweep", ["swept"]),
 ("weep", ["wept"]), ("bend", ["bent"]), ("bleed", ["bled"]), ("breed", ["bred"]), ("deal", ["dealt"]),
 ("dwell", ["dwelt", "dwelled"]), ("flee", ["fled"]), ("kneel", ["knelt", "kneeled"]), ("light", ["lit", "lighted"]),
 ("mean", ["meant"]), ("shine", ["shone", "shined"]), ("slide", ["slid"]), ("sting", ["stung"]),
 ("stink", ["stank", "stunk"]), ("tread", ["trod"]), ("weave", ["wove"]), ("wring", ["wrung"]),
]

TASKS = {"antonym": ANTONYM, "capital": CAPITAL, "past_tense": PAST_TENSE}


def get_task(name, n_train=16, seed=0):
    """Deterministic split into n_train demonstrations and the remaining test items.
    Inputs are deduplicated (keep first occurrence)."""
    import random
    items, seen = [], set()
    for x, ys in TASKS[name]:
        if x.lower() in seen:
            continue
        seen.add(x.lower())
        items.append((x, list(ys)))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n_train], items[n_train:]


if __name__ == "__main__":
    for k, v in TASKS.items():
        tr, te = get_task(k)
        print(f"{k}: {len(v)} items -> {len(tr)} train / {len(te)} test")


# ----------------------------------------------------------------------------- v2 task suite
EN_FR = [
 ("dog", ["chien"]), ("cat", ["chat"]), ("house", ["maison"]), ("water", ["eau"]), ("bread", ["pain"]),
 ("book", ["livre"]), ("tree", ["arbre"]), ("sun", ["soleil"]), ("moon", ["lune"]), ("star", ["étoile", "etoile"]),
 ("car", ["voiture"]), ("door", ["porte"]), ("window", ["fenêtre", "fenetre"]), ("table", ["table"]), ("chair", ["chaise"]),
 ("apple", ["pomme"]), ("milk", ["lait"]), ("cheese", ["fromage"]), ("egg", ["œuf", "oeuf"]), ("fish", ["poisson"]),
 ("bird", ["oiseau"]), ("horse", ["cheval"]), ("cow", ["vache"]), ("pig", ["cochon", "porc"]), ("sheep", ["mouton"]),
 ("flower", ["fleur"]), ("garden", ["jardin"]), ("city", ["ville"]), ("street", ["rue"]), ("school", ["école", "ecole"]),
 ("child", ["enfant"]), ("man", ["homme"]), ("woman", ["femme"]), ("boy", ["garçon", "garcon"]), ("girl", ["fille"]),
 ("father", ["père", "pere"]), ("mother", ["mère", "mere"]), ("brother", ["frère", "frere"]), ("sister", ["sœur", "soeur"]), ("friend", ["ami", "amie"]),
 ("king", ["roi"]), ("queen", ["reine"]), ("day", ["jour", "journée"]), ("night", ["nuit"]), ("morning", ["matin"]),
 ("year", ["an", "année", "annee"]), ("week", ["semaine"]), ("month", ["mois"]), ("time", ["temps"]), ("hour", ["heure"]),
 ("red", ["rouge"]), ("blue", ["bleu"]), ("green", ["vert"]), ("black", ["noir"]), ("white", ["blanc"]),
 ("yellow", ["jaune"]), ("big", ["grand", "gros"]), ("small", ["petit"]), ("hot", ["chaud"]), ("cold", ["froid"]),
 ("new", ["nouveau", "neuf"]), ["old", ["vieux", "ancien"]], ("good", ["bon"]), ("bad", ["mauvais"]), ("happy", ["heureux", "content"]),
 ("sad", ["triste"]), ("fast", ["rapide", "vite"]), ("slow", ["lent"]), ("strong", ["fort"]), ("weak", ["faible"]),
 ("rich", ["riche"]), ("poor", ["pauvre"]), ("young", ["jeune"]), ("beautiful", ["beau", "belle"]), ("easy", ["facile"]),
 ("difficult", ["difficile"]), ("open", ["ouvert", "ouvrir"]), ("closed", ["fermé", "ferme"]), ("full", ["plein"]), ("empty", ["vide"]),
 ("hand", ["main"]), ("head", ["tête", "tete"]), ("eye", ["œil", "oeil"]), ("heart", ["cœur", "coeur"]), ("foot", ["pied"]),
 ("mouth", ["bouche"]), ("nose", ["nez"]), ("ear", ["oreille"]), ("hair", ["cheveux", "cheveu"]), ("tooth", ["dent"]),
 ("fire", ["feu"]), ("earth", ["terre"]), ("sky", ["ciel"]), ("sea", ["mer"]), ("mountain", ["montagne"]),
 ("river", ["rivière", "riviere", "fleuve"]), ("forest", ["forêt", "foret"]), ("rain", ["pluie"]), ("snow", ["neige"]), ("wind", ["vent"]),
 ("love", ["amour", "aimer"]), ("war", ["guerre"]), ("peace", ["paix"]), ("life", ["vie"]), ("death", ["mort"]),
 ("money", ["argent"]), ("work", ["travail"]), ("word", ["mot"]), ("name", ["nom"]), ("world", ["monde"]),
 ("country", ["pays"]), ("language", ["langue", "langage"]), ("music", ["musique"]), ("song", ["chanson"]), ("game", ["jeu"]),
 ("to eat", ["manger"]), ("to drink", ["boire"]), ("to sleep", ["dormir"]), ("to read", ["lire"]), ("to write", ["écrire", "ecrire"]),
 ("to speak", ["parler"]), ("to see", ["voir"]), ("to go", ["aller"]), ("to come", ["venir"]), ("to give", ["donner"]),
 ("one", ["un"]), ("two", ["deux"]), ("three", ["trois"]), ("four", ["quatre"]), ("five", ["cinq"]),
 ("six", ["six"]), ("seven", ["sept"]), ("eight", ["huit"]), ("nine", ["neuf"]), ("ten", ["dix"]),
 ("yes", ["oui"]), ("no", ["non"]), ("thank you", ["merci"]), ("please", ["s'il vous plaît", "s'il vous plait", "s'il te plaît"]), ("hello", ["bonjour", "salut"]),
 ("goodbye", ["au revoir"]), ("today", ["aujourd'hui"]), ("tomorrow", ["demain"]), ("yesterday", ["hier"]), ("always", ["toujours"]),
 ("never", ["jamais"]), ("here", ["ici"]), ("there", ["là", "la"]), ("left", ["gauche"]), ("right", ["droite", "droit"]),
 ("summer", ["été", "ete"]), ("winter", ["hiver"]), ("spring", ["printemps"]), ("autumn", ["automne"]), ("cloud", ["nuage"]),
 ("shoe", ["chaussure"]), ("shirt", ["chemise"]), ("hat", ["chapeau"]), ("bed", ["lit"]), ("kitchen", ["cuisine"]),
 ("knife", ["couteau"]), ("spoon", ["cuillère", "cuillere"]), ("plate", ["assiette"]), ("glass", ["verre"]), ("bottle", ["bouteille"]),
 ("meat", ["viande"]), ("chicken", ["poulet"]), ("rice", ["riz"]), ("salt", ["sel"]), ("sugar", ["sucre"]),
 ("coffee", ["café", "cafe"]), ("tea", ["thé", "the"]), ("wine", ["vin"]), ("beer", ["bière", "biere"]), ("butter", ["beurre"]),
]
EN_FR = [(x, ys) for x, ys in EN_FR]

_WORDS = ("apple banana cherry orange grape lemon peach pear plum mango kiwi melon table chair window door house garden river "
          "mountain forest ocean island desert valley bridge castle tower market school church library museum hospital station "
          "airport harbor village city country planet galaxy rocket engine wheel hammer needle thread button pocket jacket "
          "sweater blanket pillow candle mirror basket bucket ladder shovel garden flower cactus bamboo maple willow cedar "
          "eagle falcon parrot pigeon robin sparrow turkey rabbit squirrel beaver badger otter walrus dolphin salmon trout "
          "shark whale turtle lizard spider beetle cricket butterfly silver copper iron marble granite crystal velvet cotton "
          "leather paper pencil crayon marker ruler eraser folder notebook laptop camera radio guitar violin trumpet drum "
          "piano flute cello banjo doctor teacher farmer baker butcher tailor pilot sailor soldier lawyer judge artist poet "
          "singer dancer actor writer driver waiter nurse monk pirate wizard knight dragon giant goblin fairy angel ghost "
          "winter summer autumn spring morning evening midnight sunrise shadow thunder lightning rainbow breeze frost mist "
          "coffee butter cheese honey pepper garlic onion carrot potato tomato pumpkin cabbage lettuce spinach walnut almond "
          "yellow purple orange silver golden violet scarlet crimson indigo amber ivory").split()

def _last_letter():
    seen, out = set(), []
    for w in _WORDS:
        if w in seen: continue
        seen.add(w); out.append((w, [w[-1]]))
    return out

LAST_LETTER = _last_letter()
TIMES_2 = [(str(n), [str(2 * n)]) for n in range(11, 200)]
ADD_7 = [(str(n), [str(n + 7)]) for n in range(10, 200)]
WORD_LENGTH = [(w, [str(len(w))]) for w, _ in LAST_LETTER]

TASKS.update({"en_fr": EN_FR, "last_letter": LAST_LETTER, "times_2": TIMES_2, "add_7": ADD_7, "word_length": WORD_LENGTH})

N_TRAIN, N_DEV = 16, 30

def get_split(name, seed=0, n_train=N_TRAIN, n_dev=N_DEV):
    """v2 protocol: 16 demonstrations (the only data the editor sees), 30 dev items
    (hyper-parameter / layer selection for baselines), the rest is test.

    A name of the form "a+b" is a two-task (or k-task) mixture in the one shared prompt format: the train and dev
    sets interleave the components' own splits (16 and 30 items each), and the test set is the components' test
    sets one after the other, so a slice of it scores each component (see subtask_slices)."""
    import random
    if "+" in name:
        parts = [get_split(p, seed, n_train, n_dev) for p in name.split("+")]
        seen = set()
        for tr, dv, te in parts:
            xs = {x.lower() for x, _ in tr + dv + te}
            if seen & xs: raise ValueError(f"components of {name} share inputs {sorted(seen & xs)[:5]}: the mixture is ill-posed in one prompt format")
            seen |= xs
        weave = lambda lists: [it for group in zip(*lists) for it in group] + [it for l in lists for it in l[min(len(m) for m in lists):]]
        return weave([p[0] for p in parts]), weave([p[1] for p in parts]), [it for p in parts for it in p[2]]
    items, seen = [], set()
    for x, ys in TASKS[name]:
        if x.lower() in seen: continue
        seen.add(x.lower()); items.append((x, list(ys)))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n_train], items[n_train:n_train + n_dev], items[n_train + n_dev:]


def subtask_slices(name, seed=0):
    """For a mixture "a+b": {component: slice of the mixture's test set that holds its items}; {} for a single task."""
    out, start = {}, 0
    if "+" in name:
        for p in name.split("+"):
            n = len(get_split(p, seed)[2]); out[p] = slice(start, start + n); start += n
    return out


# ----------------------------------------------------------------------------- harder tasks (multi-token / compositional)
def _gen_dates():
    import random
    rng = random.Random(7); months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    out, seen = [], set()
    while len(out) < 200:
        m = rng.randrange(12); d = rng.randrange(1, 29); y = rng.randrange(1950, 2030)
        x = f"{months[m]} {d}, {y}"
        if x in seen: continue
        seen.add(x); out.append((x, [f"{y}-{m+1:02d}-{d:02d}"]))
    return out

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
def _words(n):
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _words(n % 100))
def _gen_number_words():
    import random
    rng = random.Random(11); ns = rng.sample(range(21, 1000), 200)
    return [(_words(n), [str(n)]) for n in ns]

_CAMEL_WORDS = ("user name data base file path item count max value min size total price unit cost order id line number page index "
                "start time end date last name first name zip code phone number email address home town car model book title "
                "red color blue sky green leaf fast car slow train big house small door open window close gate read file write log").split()
def _gen_camel():
    import random
    rng = random.Random(13); out, seen = [], set()
    while len(out) < 200:
        k = rng.choice([2, 3, 3, 4]); ws = [rng.choice(_CAMEL_WORDS) for _ in range(k)]
        x = "_".join(ws)
        if x in seen: continue
        seen.add(x); out.append((x, [ws[0] + "".join(w.capitalize() for w in ws[1:])]))
    return out

def _roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    s = ""
    for v, r in vals:
        while n >= v: s += r; n -= v
    return s
def _gen_roman():
    import random
    rng = random.Random(17); ns = rng.sample(range(4, 400), 200)
    return [(_roman(n), [str(n)]) for n in ns]

def _gen_sst2():
    """SST-2 validation sentences -> 'positive'/'negative' (loaded lazily; needs `datasets`)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        ds = load_dataset("stanfordnlp/sst2", split="validation")
    except Exception:
        return []
    import random
    rng = random.Random(19); idx = list(range(len(ds))); rng.shuffle(idx)
    out = []
    for i in idx:
        s = ds[i]["sentence"].strip()
        if 20 <= len(s) <= 120 and "\n" not in s:
            out.append((s, ["positive" if ds[i]["label"] == 1 else "negative"]))
        if len(out) >= 220: break
    return out

TASKS.update({"date_iso": _gen_dates(), "number_words": _gen_number_words(), "camel_case": _gen_camel(), "roman": _gen_roman()})
_sst = _gen_sst2()
if _sst: TASKS["sst2"] = _sst
HARD_TASKS = ["date_iso", "number_words", "camel_case", "roman", "sst2"]


# ----------------------------------------------------------------------------- character-manipulation tasks (hard zero-shot, ICL-solvable at 7B+)
def _gen_reverse():
    return [(w, [w[::-1]]) for w, _ in LAST_LETTER if 4 <= len(w) <= 9]

def _pig(w):
    vowels = "aeiou"
    if w[0] in vowels: return w + "way"
    i = next((i for i, ch in enumerate(w) if ch in vowels), len(w))
    return w[i:] + w[:i] + "ay"
def _gen_pig_latin():
    return [(w, [_pig(w)]) for w, _ in LAST_LETTER if 4 <= len(w) <= 9]

TASKS.update({"reverse_word": _gen_reverse(), "pig_latin": _gen_pig_latin()})
HARD_TASKS += ["reverse_word", "pig_latin", "last_letter", "word_length"]


# ---------------------------------------------------------------- long-form task: GSM8K (pool prepared by wsp.prep_gsm8k)
from pathlib import Path as _Path
_pool = _Path(__file__).resolve().parent.parent / "data" / "gsm8k_pool.json"
if _pool.exists():
    import json as _json
    TASKS["gsm8k"] = [(d["question"], [f"{d['solution']} The answer is {d['answer']}."]) for d in _json.load(open(_pool))]
