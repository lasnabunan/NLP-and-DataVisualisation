from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')

class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True 
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight
        
        
        
text = '''maulana vahiduddina khana eka tanavagrasta chhatra, eka chintita bhai aura eka hina bhavana se grasita chhatra ke sabhi savalom ka jabava dete haim.
kripaya muje bataem ki hina bhavana ko kaise kama kiya ja sakata hai? hina bhavana eka niradhara avadharana hai| isa bhavana ki jaड़ ka karana vaha vyakti hota hai jo adarsha bhavana ki chaha mem ise vikasita kara leta hai| lekina vastavikata yaha hai ki adarshavada ko kabhi bhi prapta nahim kiya ja sakata hai| isa sansara mem adarshavada ka astitva kevala mana ke uchcha stara para hai, jabaki vastavika jivana mem yaha vyavaharikata mem kama karata hai| agara apa isa sacha ka pata laga lete haim to apa turanta hi isa hina bhavana se khuda ko mukta kara lenge|
meri bahana hamesha udasa rahati hai aura hara vyakti ke bare mem nakaratmaka sochati hai| maim use kaise sambhalu? samanya taura para, maim yaha kaha sakata hum ki jyadatara loga sochate haim ki unhem eka adarsha jivana chahie| eka aurata eka adarsha pati aura adarsha parivara chahati hai aura eka pati ki chahata eka adarsha patni aura parivara ki hoti hai, jo isa sansara mem sambhava nahim hai| agara prakriti ke niyama ko samaja liya jaye to jivana mem kabhi koi tanava nahim hoga| mainne akhabara mem paढ़a tha ki shaharukha khana jaba amerika ki yatra kara rahe the to unhem eyaraporta para do ghante taka roka gaya aura cheka kiya gaya| logom ne isa bata para shora machaya aura kaha ki amerikana ko isake lie maphi mangani chahie| maim sochata hum ki shaharukha khana aura baki anya logom ko isa bata ko isa taraha dekhana chahie: yaha kevala do ghante ke lie hua tha lekina jalda hi vo dina najadika a raha hai jaba sabakuchha ke lie meri bhi jancha hogi| yaha saba bhavishya mem sabake lie hoga, agara aisa sochate to hama yaha saba jalda hi bhula jate| yaha isaliye hota hai kyonki loga eka nakaratmaka sabaka sikhate haim| eka sakaratmaka vyakti hi aisi ghatanaom se eka sakaratmaka sabaka sikha sakata hai|
hamem bataya jata hai ki borda pariksha chhatrom ke lie nirnayaka hote haim lekina dara aura anishchitata muje pareshana karati hai| ina bhavanaom ne muje bhavishya ke prati ashankita bhi kara diya hai| mere kuchha dostom ka kahana hai ki agara unaka pradarshana achchha nahim raha to vo atmahatya kara lenge| mere anubhava ke anusara, isa taraha ka kuchha bhi jivana mem nirnayaka nahim hota| jivana mem sabase mahatvapurna imanadari aura driढ़ta hoti hai| mata-pita ko bachchom mem isa bhavana ko baढ़ava dena chahie| kisi pariksha mem chunava hona achchha hai lekina kevala kisi shikshana sansthana mem pravesha pane ke lie, bhavishya mem eka achchha jivana pane ke lie yaha jaruri nahim hai| mata-pita ko apane bachchom ko jivana ke siddhantom ke bare mem avashya shikshita karana chahie aura koshisha karani chahie ki unake andara dhairya, buddhimatta aura samanjasya jaise gunom ka vikasa ho| unhem bachchom ko yatharthavadi socha ki mahatta ko samajane mem madada karani chahie| unhem avashya sikhana chahie ki asaphalata se kaise sikhem aura yaha janem ki jivana mem sabakuchha pana mayane nahim rakhata hai| vastavika jivana mem kai aisi chijem hoti haim jo akele hi upayogi hoti hai| agara apa pariksha mem taॉpa karate haim to isa bata ki koi garanti nahim ki apa jivana mem bhi taॉpa sthana para hom| agara apa jivana ke siddhanta ke prati jagaruka haim to saphalata apaka anusarana jarura karegi| apane bataya ki apake dosta atmahatya ke bare mem bata karate haim, to apa jana lem ki atmahatya koi vikalpa nahim hai| apani kshamataom ko samajem aura phira apako mahasusa hoga ki atmahatya karane ke bare mem sochana khuda ko aura nirmata ko kamatara ankane ke samana hai|
muje lagata hai ki mera eka alaga vyaktitva hai jisane eka sakhta sanrakshaka ki taraha muje apane vasha mem kara rakha hai| apane dostom ki barabari mem ane ka dabava muje pareshana karata hai aura isa karana maim khuda ko ekagrita nahim kara pata| maim isa samasya se kaise nipatum? apane dostom ko adarsha samajakara unase spardha na karem| hajarom aisi kitabem haim jo saphala vyaktiyom ke jivana para charcha karati hai| apako ina kitabom ko paढ़na chahie aura aise logom ke jivana se kuchha sabaka sikhane ki koshisha karani chahie|
paढ़ai ka bahuta jyada tanava muje narvasa kara raha hai| kabhi-kabhi to maim bilkula khali ho jata hum| maim tutane ke kagara para hum| maim chahata hum ki mere parivaravale mujase aura adhika batachita kare lekina unhonne yaha kahakara muje akela chhoड़ diya hai ki aisa karane se meri paढ़ai mem khalala paड़egi| mainne ghara se dura bhagane taka ke bare mem socha liya hai...
apaka parivara snehavasha aisa kara raha hai| vastava mem vo apako adhika samaya de rahe haim taki apa khuda ki kshamataom ko talasha sakem| phira yaha shikayata kyom? ise eka avasara samajem| apaki kitabem apaki sabase achchhi dosta hai| khuda ke bharose jina sikhem| aisa karane para apa dusarom ke prati shikayata nahim hogi| apake isa vyavahara ka dusara nama yaha hai ki apa khuda ko kamatara samaja rahe haim| saphala vyakti ke jivana ke bare mem janana utana hi jaruri hai jitana parivara ke sadasyom ke samparka mem rahana| purvaja apake jivana ka margadarshana karate haim jabaki shabda apako kevala bhavanatmaka santushti dete haim| mata-pita, dosta aura skula se age bhi eka sansara hai| apako isa sansara ke bare mem aura adhika sikhane ki koshisha karani chahie| aura usa vyakti ka shukriya ada karem jo apase milane kama ate haim taki apa khuda ko samaya de sakem|
'''
​
tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
tr4w.get_keywords(10)