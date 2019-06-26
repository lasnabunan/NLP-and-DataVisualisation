from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import re


def gen_freq(text):
   
    word_list= []
    word_list= word_tokenize(text)
    print(word_list)  
    filtered_list = set_stopwords(word_list)
    print(filtered_list)
    word_freq = pd.Series(word_list).value_counts()
    filtered_freq = pd.Series(filtered_list).value_counts()
    
    
    word_freq[:20]
    filtered_freq[:20]
    
    
def set_stopwords(word_list):  
        """Set stop words"""
        filtered_list=[]
        word_s =['?', '!', '|', '.', ',', ';', '@', '-', 'andar', 'ata', 'adi', 'apa', 'apana', 'apani', 'apani', 'apane', 'abhi', 'abhi',  'adi', 'apa', 'inhim', 'inhem', 'inhom', 'itayadi', 'ityadi', 'ina', 'inaka', 'inhim', 'inhem', 'inhom', 'isa', 'isaka', 'isaki', 'isaki', 'isake', 'isamem', 'isi', 'isi', 'ise', 'unhim', 'unhem', 'unhom', 'una', 'unaka', 'unaki', 'unaki', 'unake', 'unako', 'unhim', 'unhem', 'unhom', 'usa', 'usake', 'usi', 'usi', 'use', 'eka', 'evam', 'esa', 'ese', 'aise', 'ora', 'aura', 'kai', 'kai', 'kara', 'karata', 'karate', 'karana', 'karane', 'karem', 'kahate', 'kaha', 'ka', 'kaphi', 'kai', 'ki', 'kinhem', 'kinhom', 'kitana', 'kinhem', 'kinhom', 'kiya', 'kira', 'kisa', 'kisi', 'kisi', 'kise', 'ki', 'kuchha', 'kula', 'ke', 'ko', 'koi', 'koi', 'kona', 'konasa', 'kauna', 'kaunasa', 'gaya', 'ghara', 'jaba', 'jaham', 'jaham', 'ja', 'jinhem', 'jinhom', 'jitana', 'jidhara', 'jina', 'jinhem', 'jinhom', 'jisa', 'jise', 'jidhara', 'jesa', 'jese', 'jaisa', 'jaise', 'jo', 'taka', 'taba', 'taraha', 'tinhem', 'tinhom', 'tina', 'tinhem', 'tinhom', 'tisa', 'tise', 'to', 'tha', 'thi', 'thi', 'the', 'dabara', 'davara', 'diya', 'dusara', 'dusare', 'dusare', 'do', 'dvara', 'na', 'nahim', 'nahim', 'na', 'niche', 'nihayata', 'niche', 'ne', 'para', 'pahale', 'pura', 'pura', 'pe', 'phira', 'bani', 'bani', 'bahi', 'bahi', 'bahuta', 'bada', 'bala', 'bilakula', 'bhi', 'bhitara', 'bhi', 'bhitara', 'magara', 'mano', 'me', 'mem', 'yadi', 'yaha', 'yaham', 'yaham', 'yahi', 'yahi', 'ya', 'yiha', 'ye', 'rakhem', 'ravasa', 'raha', 'rahe', 'vasa', 'lie', 'liye', 'lekina', 'va', 'vageraha', 'varaga', 'varga', 'vaha', 'vaham', 'vaham', 'vahim', 'vahim', 'vale', 'vuha', 've', 'va', 'airaha', 'sanga', 'sakata', 'sakate', 'sabase', 'sabhi', 'sabhi', 'satha', 'sabuta', 'sabha', 'sara', 'se', 'so', 'hi', 'hi', 'hua', 'hua', 'hui', 'hui', 'hue', 'he', 'hem', 'hai', 'haim', 'ho', 'hota', 'hoti', 'hoti', 'hote', 'hona', 'hone', 'sabha', 'sara', 'se', 'so', 'hi', 'hi', 'hua', 'hua', 'hui', 'hui', 'hue', 'he', 'hem', 'hai', 'haim', 'ho', 'hota', 'hoti', 'hoti', 'hote', 'hona', 'hone''sabha', 'sara', 'se', 'so', 'hi', 'hi', 'hua', 'hua', 'hui', 'hui', 'hue', 'he', 'hem', 'hai', 'haim', 'ho', 'hota', 'hoti', 'hoti', 'hote', 'hona', 'hone']
        print(word_list)
        for w in word_list:
            if w not in word_s:
                filtered_list.append(w)
                print(w)
        return filtered_list
                
           

text = '''maulana vahiduddina khana eka tanavagrasta chhatra , eka chintita bhai aura eka hina bhavana se grasita chhatra ke sabhi savalom ka jabava dete haim .
kripaya muje bataem ki hina bhavana ko kaise kama kiya ja sakata hai ? hina bhavana eka niradhara avadharana hai | isa bhavana ki jada ka karana vaha vyakti hota hai jo adarsha bhavana ki chaha mem ise vikasita kara leta hai | lekina vastavikata yaha hai ki adarshavada ko kabhi bhi prapta nahim kiya ja sakata hai | isa sansara mem adarshavada ka astitva kevala mana ke uchcha stara para hai , jabaki vastavika jivana mem yaha vyavaharikata mem kama karata hai | agara apa isa sacha ka pata laga lete haim to apa turanta hi isa hina bhavana se khuda ko mukta kara lenge |
meri bahana hamesha udasa rahati hai aura hara vyakti ke bare mem nakaratmaka sochati hai | maim use kaise sambhalu ? samanya taura para, maim yaha kaha sakata hum ki jyadatara loga sochate haim ki unhem eka adarsha jivana chahie | eka aurata eka adarsha pati aura adarsha parivara chahati hai aura eka pati ki chahata eka adarsha patni aura parivara ki hoti hai, jo isa sansara mem sambhava nahim hai | agara prakriti ke niyama ko samaja liya jaye to jivana mem kabhi koi tanava nahim hoga | mainne akhabara mem paढ़a tha ki shaharukha khana jaba amerika ki yatra kara rahe the to unhem eyaraporta para do ghante taka roka gaya aura cheka kiya gaya | logom ne isa bata para shora machaya aura kaha ki amerikana ko isake lie maphi mangani chahie | maim sochata hum ki shaharukha khana aura baki anya logom ko isa bata ko isa taraha dekhana chahie : yaha kevala do ghante ke lie hua tha lekina jalda hi vo dina najadika a raha hai jaba sabakuchha ke lie meri bhi jancha hogi | yaha saba bhavishya mem sabake lie hoga, agara aisa sochate to hama yaha saba jalda hi bhula jate | yaha isaliye hota hai kyonki loga eka nakaratmaka sabaka sikhate haim| eka sakaratmaka vyakti hi aisi ghatanaom se eka sakaratmaka sabaka sikha sakata hai |
hamem bataya jata hai ki borda pariksha chhatrom ke lie nirnayaka hote haim lekina dara aura anishchitata muje pareshana karati hai | ina bhavanaom ne muje bhavishya ke prati ashankita bhi kara diya hai | mere kuchha dostom ka kahana hai ki agara unaka pradarshana achchha nahim raha to vo atmahatya kara lenge | mere anubhava ke anusara, isa taraha ka kuchha bhi jivana mem nirnayaka nahim hota | jivana mem sabase mahatvapurna imanadari aura driढ़ta hoti hai | mata-pita ko bachchom mem isa bhavana ko baढ़ava dena chahie | kisi pariksha mem chunava hona achchha hai lekina kevala kisi shikshana sansthana mem pravesha pane ke lie, bhavishya mem eka achchha jivana pane ke lie yaha jaruri nahim hai | mata-pita ko apane bachchom ko jivana ke siddhantom ke bare mem avashya shikshita karana chahie aura koshisha karani chahie ki unake andara dhairya, buddhimatta aura samanjasya jaise gunom ka vikasa ho | unhem bachchom ko yatharthavadi socha ki mahatta ko samajane mem madada karani chahie | unhem avashya sikhana chahie ki asaphalata se kaise sikhem aura yaha janem ki jivana mem sabakuchha pana mayane nahim rakhata hai | vastavika jivana mem kai aisi chijem hoti haim jo akele hi upayogi hoti hai | agara apa pariksha mem taॉpa karate haim to isa bata ki koi garanti nahim ki apa jivana mem bhi taॉpa sthana para hom | agara apa jivana ke siddhanta ke prati jagaruka haim to saphalata apaka anusarana jarura karegi | apane bataya ki apake dosta atmahatya ke bare mem bata karate haim, to apa jana lem ki atmahatya koi vikalpa nahim hai | apani kshamataom ko samajem aura phira apako mahasusa hoga ki atmahatya karane ke bare mem sochana khuda ko aura nirmata ko kamatara ankane ke samana hai |
muje lagata hai ki mera eka alaga vyaktitva hai jisane eka sakhta sanrakshaka ki taraha muje apane vasha mem kara rakha hai | apane dostom ki barabari mem ane ka dabava muje pareshana karata hai aura isa karana maim khuda ko ekagrita nahim kara pata | maim isa samasya se kaise nipatum ? apane dostom ko adarsha samajakara unase spardha na karem | hajarom aisi kitabem haim jo saphala vyaktiyom ke jivana para charcha karati hai | apako ina kitabom ko paढ़na chahie aura aise logom ke jivana se kuchha sabaka sikhane ki koshisha karani chahie |
padhai ka bahuta jyada tanava muje narvasa kara raha hai | kabhi-kabhi to maim bilkula khali ho jata hum | maim tutane ke kagara para hum | maim chahata hum ki mere parivaravale mujase aura adhika batachita kare lekina unhonne yaha kahakara muje akela chhoड़ diya hai ki aisa karane se meri paढ़ai mem khalala paड़egi | mainne ghara se dura bhagane taka ke bare mem socha liya hai...
apaka parivara snehavasha aisa kara raha hai | vastava mem vo apako adhika samaya de rahe haim taki apa khuda ki kshamataom ko talasha sakem | phira yaha shikayata kyom ? ise eka avasara samajem | apaki kitabem apaki sabase achchhi dosta hai | khuda ke bharose jina sikhem | aisa karane para apa dusarom ke prati shikayata nahim hogi | apake isa vyavahara ka dusara nama yaha hai ki apa khuda ko kamatara samaja rahe haim | saphala vyakti ke jivana ke bare mem janana utana hi jaruri hai jitana parivara ke sadasyom ke samparka mem rahana | purvaja apake jivana ka margadarshana karate haim jabaki shabda apako kevala bhavanatmaka santushti dete haim | mata-pita, dosta aura skula se age bhi eka sansara hai | apako isa sansara ke bare mem aura adhika sikhane ki koshisha karani chahie | aura usa vyakti ka shukriya ada karem jo apase milane kama ate haim taki apa khuda ko samaya de sakem |
'''

gen_freq(text)

text = re.sub('[?!.;:,#@-|]',"", text)