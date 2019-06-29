from collections import OrderedDict
import numpy as np
import spacy
import pandas as pd
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
        word_s =[' - ', 'jaenge|', 'haim|', '"', 'hai)|', 'paenge|', '\n', '(', ')' ,'-', 'hai|','?', '!', ',', '|', '.', ';', ':', 'andar', 'ata', 'adi', 'apa', 'apana', 'apani', 'apani', 'apane', 'abhi', 'abhi',  'adi', 'apa', 'inhim', 'inhem', 'inhom', 'itayadi', 'ityadi', 'ina', 'inaka', 'inhim', 'inhem', 'inhom', 'isa', 'isaka', 'isaki', 'isaki', 'isake', 'isamem', 'isi', 'isi', 'ise', 'unhim', 'unhem', 'unhom', 'una', 'unaka', 'unaki', 'unaki', 'unake', 'unako', 'unhim', 'unhem', 'unhom', 'usa', 'usake', 'usi', 'usi', 'use', 'eka', 'evam', 'esa', 'ese', 'aise', 'ora', 'aura', 'kai', 'kai', 'kara', 'karata', 'karate', 'karana', 'karane', 'karem', 'kahate', 'kaha', 'ka', 'kaphi', 'kai', 'ki', 'kinhem', 'kinhom', 'kitana', 'kinhem', 'kinhom', 'kiya', 'kira', 'kisa', 'kisi', 'kisi', 'kise', 'ki', 'kuchha', 'kula', 'ke', 'ko', 'koi', 'koi', 'kona', 'konasa', 'kauna', 'kaunasa', 'gaya', 'ghara', 'jaba', 'jaham', 'jaham', 'ja', 'jinhem', 'jinhom', 'jitana', 'jidhara', 'jina', 'jinhem', 'jinhom', 'jisa', 'jise', 'jidhara', 'jesa', 'jese', 'jaisa', 'jaise', 'jo', 'taka', 'taba', 'taraha', 'tinhem', 'tinhom', 'tina', 'tinhem', 'tinhom', 'tisa', 'tise', 'to', 'tha', 'thi', 'thi', 'the', 'dabara', 'davara', 'diya', 'dusara', 'dusare', 'dusare', 'do', 'dvara', 'na', 'nahim', 'nahim', 'na', 'niche', 'nihayata', 'niche', 'ne', 'para', 'pahale', 'pura', 'pura', 'pe', 'phira', 'bani', 'bani', 'bahi', 'bahi', 'bahuta', 'bada', 'bala', 'bilakula', 'bhi', 'bhitara', 'bhi', 'bhitara', 'magara', 'mano', 'me', 'mem', 'yadi', 'yaha', 'yaham', 'yaham', 'yahi', 'yahi', 'ya', 'yiha', 'ye', 'rakhem', 'ravasa', 'raha', 'rahe', 'vasa', 'lie', 'liye', 'lekina', 'va', 'vageraha', 'varaga', 'varga', 'vaha', 'vaham', 'vaham', 'vahim', 'vahim', 'vale', 'vuha', 've', 'va', 'airaha', 'sanga', 'sakata', 'sakate', 'sabase', 'sabhi', 'sabhi', 'satha', 'sabuta', 'sabha', 'sara', 'se', 'so', 'hi', 'hi', 'hua', 'hua', 'hui', 'hui', 'hue', 'he', 'hem', 'hai', 'haim', 'ho', 'hota', 'hoti', 'hoti', 'hote', 'hona', 'hone', 'sabha', 'sara', 'se', 'so', 'hi', 'hi', 'hua', 'hua', 'hui', 'hui', 'hue', 'he', 'hem', 'hai', 'haim', 'ho', 'hota', 'hoti', 'hoti', 'hote', 'hona', 'hone''sabha', 'sara', 'se', 'so', 'hi', 'hi', 'hua', 'hua', 'hui', 'hui', 'hue', 'he', 'hem', 'hai', 'haim', 'ho', 'hota', 'hoti', 'hoti', 'hote', 'hona', 'hone']

        for word in word_s:
           lexeme = nlp.vocab[word]
           lexeme.is_stop = True 
            
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                #if token.pos_ in candidate_pos and token.is_stop is False:
                if token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
            
       # print(sentences)    
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
        token_pairs_freq = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])    
                    token_pairs.append(pair)
        #for word in token_pairs:
         #   print(word)
        unique_values = []
        unique_values = pd.Series(token_pairs).unique()
        #for i in unique_values:
            #print(i)
            
        arr = pd.DataFrame([])
       
        ls = []
        for i in unique_values:
            count = 0
            for j in token_pairs:
                if i == j:
                    count = count + 1
                else:
                    continue
            #print(i, count)
            lm = []
            lm = [i,count]
            ls.append(lm)
            #arr = arr.append(pd.DataFrame({'Pairs': pd.Series(data = [i]), 'Freq': pd.Series(data = [count])}))
        
        #for word in ls:
         #   print(word)
            
        data = pd.DataFrame(ls,columns=['Pairs','Freq'])
        data = data.sort_values('Freq', ascending = False)
        for i in data['Freq']:
            print(i)
        #for i in data['Pairs']:
         #   print(i)
        print(data)
         
      
        
        #print(token_pairs_freq)
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

    
    def get_keywords(self, number):
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
        
        
        
text = '''trevarsa tainka kabutara, mora aura titara jaisi vibhinna jatiyom ke pakshiyom ke karana pakshi - avalokana ke lie prasiddha hai | guru shikhara maunta abu ki sabase unchi choti hai jaham se pure shahara ka bahuta achchha naja़ara dikhata hai| parvata ke shikhara para sthita eka chhota-sa shaiva punya-sthala aura dattatreya ka mandira bhramana ke do mahatvapurna sthana haim| rajasthana ke thara marusthala ke asali jadu ka anubhava pane ka sabase achchha tarika hai marusthala sapha़ari | maunta abu mem rajasthali, rajasthana shasakiya hastashilpa emporiyama aura khadi bhandara kharidari ke mahatvapurna sthana haim | apraila se juna aura aktubara se navambara ka samaya isa shahara ke bhramana ke lie sabase achchha hai | sahasika karyom ki talasha mem nikale paryatakom ke lie thara marusthala ke andaruni hissom mem ghumane, shaharom aura aitihasika khandaharom ko dekhane ke lie unta sapha़ari sarvottama vikalpa hai | rajasthana ke chhote- chhote gamvom mem ghumane ke anokhe anubhava ke lie unta sapha़ari eka romanchaka aura jokhima -bhara sadhana hai | thara marusthala ke khule bhubhaga para ghumane lie jipa sapha़ari eka adbhuta sadhana hai aura rajasthana ke khubasurata, shanta aura ranga-birange ilake ko dekhane ka anokha avasara pradana karati hai | ananda aura romancha ke mishrana ke satha rajasthana ke thara marusthala mem jipa sapha़ari vaham ke kuchha shanadara gantavyom ko samavishta karati hai| thara marusthala ke bhavya retile tile rajasthana ki anokhi dena haim jo ki vaham ki ranga-birangi sanskriti,vanya - jivana, aitihasika smarakom, kilom, udyanom aura jilom se alaga haim | insani hunara aura acharana se anachhue rajasthana ke ye retile tile prakriti ka eka uttama upahara haim jo registana ka eka jadui naja़ara pesha karate haim | nile asamana ke niche hava se akara mem dhale retile tile duniya -bhara se bada़i sankhya mem paryatakom ko akarshita karate haim aura rajasthana ki koi bhi saira ina retile tilom ki saira ke bina adhuri hai | retile tilom dvara bani laharem photographa़rom ke lie pha़oto khinchane ke lie uttama sthana rachati haim | rajasthana ke marusthala ki saira karane ka sarvottama tarika unta sapha़ari ke madhyama se hai, jo registani paryatana ka sabase lokapriya sadhana hai | rajasthana ke andaruni hisse sirpha़ isi sadhana ke jarie dekhe ja sakate haim jise prayah ‘registana ka jahaja’ kaha jata hai | rajasthana mem unta sapha़ari solahavim shatabdi se prachalana mem hai jise taba mukhya rupa se samana dhone ke lie istemala kiya jata tha, aura aba yaha manoranjana ka mahatvapurna sadhana hai | rajasthana ke thara marusthala mem chalane vale unta sapha़ari ke aja ke rupa mem prachina unta- karavam ka jadu dekha ja sakata hai | rajasthana - bhramana ka eka mahatvapurna hissa, unta sapha़ari paramparika aura ranga-birange rajasthana ki andaruni janki dikhata hai |
jaisalamera, jodhapura aura bikanera retile tilom ke lie jane jate haim para inamem se sabase lokapriya jaisalamera hai | adhikatara reta ke tile jaisalamera ke asapasa haim | saima sainda dyunsa eka uttama paryataka sthana hai jo suryasta evam suryodaya ke darshaniya sthala ke rupa mem jana jata hai | saima sainda dyunsa manava ke lie prakriti ke sarvashreshtha upaharom mem se eka hai | jaisalamera ke thara marusthala kshetra mem sthita saima sainda dyunsa sabase adhika prasiddha reta ke tilom mem se hai jo sala -bhara paryatakom ki bahuta bada़i sankhya ko akarshita karata hai | manavara marusthala bahari sahasika karanamom evam romancha ke lie sarvottama hai | paryataka apane apa ko adivasi gamvom ki yatra, marusthala ke vanya jivana ka darshana aura sthaniya shilpakarom ke shilpa ko dekhane mem vyasta rakha sakate haim | bikanera rajasthana ke uttara mem sthita hai aura samanyataya unta- pradesha ke rupa mem jana jata hai | vishva ki dasa pramukha shanadara relagada़iyom mem se eka, pailesa ऑna hvilsa, bharata ki shana kahi ja sakati hai jo yatra ki apani anupama, atulaniya, avismaraniya evam advitiya shaili ke lie jani jati hai | rajasthana ki rajasi bhumi ke darshana karane ke sarvottama madhyamom mem se eka, pailesa ऑna vhilsa bharata ke preranadayaka, atulaniya, avishvasaniya, prabhavashali evam preranatmaka kshetra se gujarati hai | bharatiya rela evam rajasthana paryataka vikasa nigama ke sanyukta prayasom se bharata ki pahali avakasha - relagada़i, pailesa ऑna vhilsa ne 1982 mem apani rajasi yatra arambha ki thi| aja yaha bhavya pailesa ऑna vhilsa relagada़i rajasi rajasthana ka anivarya hissa aura bharata ka gaurava ho gai hai jo rajasthana ke kuchha sarvottama paryataka - sthalom ke darshana karati hai | yaha rajasi yatra mai se agasta ke bicha ke mahinom ko chhoda़kara varsha - bhara hoti hai | vishva ki sarvottama romanchaka yatraom mem se eka pailesa ऑna vhilsa yatra ke daurana vishva stara ki sevaem evam suvidhaem pradana karati hai yaha rajasi yatra dilli se arambha hoti hai aura jayapura - jaisalamera - jodhapura - savai madhopura- chittauda़gadha़ - udayapura - bharatapura - agara hote hue dilli mem akara samapta hoti hai | pushkara mela pratyeka varsha hindu maha kartika (aktubara - navambara) ki shubha purnima ke avasara para lagata hai | rajasthana - bhavya, prakritika aura vastushilpiya ashcharyom se vibhushita pradesha | alabarta hala sangrahalaya jayapura, rajya ka sabase purana sangrahalaya mana jata hai |
paryatana vibhaga ne paryatana sankula ke nirmana ke lie kai nae sthanom ko nirdeshita kiya hai | hariyana paryatana ne hariyana ke sabhi kshetrom mem hotalom, motalom aura bhojanalayom ka prabandha kiya hai | panipata yamuna nadi ke tata para sthita hai aura yaham panipata ke tina aitihasika yuddha lada़e gae the | panipata eka audyogika shahara hai aura hathakaragha utpadom ke lie jana jata hai | jaham taka pata laga hai ‘danavira karna ka shahara’ karanala, divarom se ghira hua shahara raha hai aura shayada kisi samaya mem eka durga raha hoga| karanala jutom, krishi anusandhana sansthanom aura basamati chavala ke lie prasiddha hai|
panchakula jile mem uposhnakatibandhiya mahadvipiya manasuni jalavayu hai jaham hamem ritu avartana, garama grishmaritu, thandi shitaritu, aniyamita varsha aura tapamana mem atyadhika parivartana milata hai|
paryatana karane ke lie atyanta manorama sthalom mem se kuchha haim, kinnaura jile mem sangala ghati aura kalpa, shimala jile mem naladehara, narakanda aura sarahana, kullu jile mem manali aura manikarana, kangada़a jile mem dharmashala tatha lahaula aura spiti mem tabo | palamapura apane krishi mahavidyalaya aura chaya baganom ke lie prasiddha hai | himachala pradesha vanya-jivana premiyom, pakshi prekshakom tatha sahasika paryatakom ke saira karane ke lie bahuta achchhi jagaha hai | shimala paryatakom mem eka lokapriya shahara hai | shimala ki saira karane ke lie sala ke sarvottama mahine marcha - juna (vasanta), sitambara - aktubara (sharada) aura disambara - janavari (shitakalina himapata ka anubhava lene ke lie) haim | himachchhadita himalaya shrinkhala shimala se dikhai pada़ti hai | rajasthana apane paryatakom ke lie vividha prakara ke bhojana prastuta karata hai- yaha vividhata isaki sanskriti, isaki paitrika sanrachana, isaki bhaugolika sthiti aura isaki jalavayu jaisi vividha hai | da raॉyala kairija़ma ऑpha rajasthana tura (rajasi akarshana ki rajasthana yatra) ise sansara ke ati vanchhita paryataka gantavya karara deta hai | bharata ki prakritika sampada usaki sanskritika sampada ki taraha sanriddha aura vividha hai | yaham ke vanya-jiva abhayaranya aura rashtriya udyana apako vahi sara dete haim | raॉyala bangala taigara isa sanriddha kshetra mem basa hai, (jo) raॉyala taigara kaita ka ekamatra nivasa-sthana (hai)| bangala taigara sabase teja़ dauda़ne vala janavara hai | agara apa raॉyala bangala taigara mem dilachaspi rakhate haim tatha use usake svabhavika parivesha mem dekhana chahate haim, (to) ina bagha rakshita sthanom ki yatra avashya anandadayaka hogi | karbeta ka karbeta neshanala parka baghom aura satha hi unake shikara ke lie bhi ekaashraya-sthala hai, jinamem chara prakara ke hirana, jangali suara aura kuchha kama prasiddha janavara sammilita haim | svarna mandira sikkhom ke pavitra nagara anritasara mem sthita hai, jo kabhi ghana jangala tha aura gurunanaka ka nivasa-sthala tha | maidanom ke upara sthita dharmashala ghane chida़ ke peda़om aura devadara ke jangalom se ghiri hai| bahuta si jala dharaom ke satha pasa ki hima-rekha aura shitala svastha vatavarana parivesha ko bahuta hi lubhavana banata hai| kabhi shasana ka eka mahatvapurna kendra mana jane vala chanda vansha ka rajadhani shahara kangada़a, gaurava ki eka gatha kahata hai, jo itihasa mem kho gaya hai| ninna himalaya ki manorama ghatiyom mem se eka, utkrishta dhauladhara pahadiyom dvara parirakshita yaha ghati hari-bhari aura sanriddha hai | sikkima ke parvatiya kshetra mem hare-bhare pahada़i rastom, stupom, mathom aura mandirom ke satha-satha romancha ki prachurata hai | sanriddha sanskriti aura parampara ka parichaya prapta kijie jo ki apane asimita anandaprada sanskarom evam utsavom ke karana purnatah vilakshana evam mohaka hai | sikkima eka sukhadayi yatra pradana karata hai – romancha aura khoja ka eka paryatana | bahuta lambe samaya se manava apane hridaya mem eka chida़iya ki bhanti vishala nile akasha mem uda़ne ki gahari chaha lie hue hai, kuchha hada taka hama manava khule akasha mem uda़ne ke lie havai - khela ke romancha ko apanate hue apani isa pyasa ko buja pane mem sakshama haim | 26,000 phuta ki unchai para barpha se dhaki chotiyom ke shikhara vala himalaya vishva ki sarvashreshtha parvata shrrinkhala hai|
bharata mem paidala - yatra eka avismaraniya anubhava hai kyom ki yaha apako prakriti dvara nirmita ajnata rastom ka anubhava hi nahim deti,(balki) paidala - yatra paryataka ko desha aura isake logom ke satha sidhe samparka mem lati hai aura apako parvatiya vatavarana ko samajane mem sahayata karati hai | himalaya vishva ki sabase taruna parvata shrrinkhalaom mem se eka hai aura vishva ke sarvashreshtha parvatarohana kshetrom mem se gina jata hai| bharatiya himalaya shrrinkhala nissandeha vishva ki sabase bhavya aura prabhavashali parvata shrrinkhalaom mem se eka hai| thara registana mem sahasika yatra aisi yatra hai jise apa apane pure jivana mem nahim bhula paenge| paryataka usi registani anubhava ka ananda jaisalamera, rajasthana mem saima sainda dyunsa mem bhi le sakate haim jaham paryatakom ke manoranjana ke lie rajasthana paryatana vibhaga dvara sayankala mem sanskritika pradarshanom ka ayojana kiya jata hai. gira rashtriya udyana sagauna, gulamohara, khaira aura baragada vrikshom ka eka mila-jula patajada़i vana hai| satha hi gira rashtriya udyana bharata ke kisi bhi udyana ki tulana mem chite ki sabase adhika abadi vala avasa hai| bharata mem bahuta se bagha arakshita kshetra haim, jo isa khunkhara janavara ko surakshita rakha rahe haim, lekina jitani bahutayata aura niyamitata se inhem kanha rashtriya udyana mem dekha ja sakata hai vaisa aura kahim nahim | vishva ke sarvottama pakshi udyanom mem se eka, bharatapura pakshi abhayavana (kevaladeva ghana neshanala parka) eka arakshita kshetra hai jo pashu-varga ko bhi sanrakshana deta hai| udyana dekhane vale adhikansha paryatakom ka mukhya akarshana bahuta se pravasi pakshi hote haim jo saiberiya aura madhya eshiya jaise sudura pradeshom se ate haim aura apane prajanana sthanom ko lautane se pahale ve apani thanda bharatapura mem bitate haim | bharatapura abhayavana ke bada bharatapura shasakiya sangrahalaya mukhya dhyana khinchane valom mem se eka hai, jo bharatapura ke bhutakalina rajasi vaibhava ki jalaki pradana karata hai | sariska udyana tendue, jangali kutte, vana billi, lakada़bagghe, bheda़iye aura bagha ke satha-satha bahuta se mansahari pashuom ka ghara hai| sariska risasa bandarom ki adhika janasankhya ke lie suprasiddha hai, jo ki talavriksha ke asa-pasa adhika sankhya mem paye jate hai|
ranathambhaura rashtriya udyana projekta taigara rija़rvsa ऑpha da varlda (vishva ki bagha sanrakshana pariyojana) ke antargata sammilita hai aura apani bagha janasankhya ke lie adhika lokapriya hai| bagha ko usake prakritika avasa mem dekhana pratyeka vanyajivana samarthaka ki sabase vanchhaniya kalpana hai, jo ki ranathambhaura ke vana mem purna ho sakati hai| ranathambhaura vanyajivana udyana vishva ke kuchha una darshaniya sthanom mem se hai jaham baghom ka varchasva hai| ranathambhaura vanyajivana udyana daladali magaramachchha aura ubhayachara pashuom ke satha-satha chida़iyom ki 272 jatiyom aura sarisripom ka eka avasa hai| ranathambhaura vana ke chhupe hue khajane ki khoja ke lie jipa yatra bhi upalabdha karata hai, kyonki yaha raॉyala taigara ki khoja ke lie sarvottama sadhana hai| vanyajivana rashtriya udyana mem anya akarshana ranathambhaura kile ke khandahara aura jogi mahala hai jo ki 10vim shatabdi ko dinankita karate hai| aba vana vishrama griha mem parivartita jogi mahala bharata mem dusare sabase bada़e baragada ke vriksha ke lie prasiddha hai| taja nissandeha vishva ke sabase bhavya bhavanom mem se eka hai| apani sthapatya bhavyata aura kalatmaka sundarata ke lie prasiddha taja manava ki sabase gauravashali rachanaom mem gina jata hai aura vishva ke shreshtha ashcharyom ki suchi mem nirapavada rupa se sammilita hai| bharatiya pratishthita kavi taigora dvara " shashvatata ke chehare para ansu" ke rupa mem varnita, tajamahala nissandeha mugala vastukala ki parakashtha hai aura nitanta saralata se vishva ke ati ashcharyajanaka bhavanom mem se eka hai| agara apani sthapatya bhavyata ki sanriddhi, baja़arom aura abhushanom ke satha vishva ke ati utkrishta shaharom mem se hai| taja mahotsava ke samaya shilpagrama shahara mem pure vishva ke paryataka ekatra hote haim aura isaka bharapura ananda lete haim| agara ane vale paryataka taja ko apani yadom mem banae rakhane ke lie apane satha isaka eka namuna avashya lem jaenge| agara mem chamada़e ke thailom, kasidakari vale jutom ki kharidadari ka bhi bada़a akarshana hai aura petha (mithai) to nishchita rupa se agara ki eka visheshata hai| yadyapi sudura mem sthita hai, phira bhi khajuraho mandira sankula sthala videshi aura bharatiya donom paryatakom ke lie sarvadhika lokapriya sthanom mem se eka hai| khajuraho ke mandira apani murti kala ke satha paryataka ka dhyana banae rakhate haim, jo ki bahuta utkarsha aura jatila hai, jise aba koi bhi hubahu banane ka sapana bhi nahim dekha sakata| nishpadana mem shreshtha aura bhavabhivyakti mem utkrishta khajuraho mandira naritva ko samarpana haim| kalakara ki srijanatmaka pravrittiyom ne jivana ke vibhinna pahaluom aura bhavom ko patthara mem sundarata se utara hai|'''
tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=2, lower=False)
tr4w.get_keywords(25)