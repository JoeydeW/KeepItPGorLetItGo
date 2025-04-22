import string
from string import punctuation
import re
import nltk
#When running for the first time, run in a python console:
# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')

import pandas as pd
import numpy as np
import sklearn
import json
import pickle
import json
import os
from nltk.tokenize import SyllableTokenizer
import sys
import ssl
import pathlib
from pathlib import Path

def get_POS(row):

    retList = []

    for tag in nltk.pos_tag(row):
        retList.append(tag[1])

    return retList

SSP = SyllableTokenizer()
def magic_e(word):
    result = SSP.tokenize(word)
    syll_count = len(result)
    if syll_count == 1:
        return syll_count
    if re.search('e$', result[len(result) - 1]):
        modified = ''.join([result[i] for i in [len(result) - 2, len(result) - 1]])
        result[len(result) - 2] = modified
        del result[len(result) - 1]
        syll_count = len(result)
    return syll_count

SSP = SyllableTokenizer()
def magic_e_result(word):
    result = SSP.tokenize(word)
    if re.search('e$', result[len(result) - 1]):
        modified = ''.join([result[i] for i in [len(result) - 2, len(result) - 1]])
        result[len(result) - 2] = modified
        del result[len(result) - 1]
    return result

lines = []
import csv

#https://github.com/open-dict-data/ipa-dict
# import pandas as pd
# file_path = 'en_US.txt'
# data_list = []
# with open(file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         line = line.strip()
#         key, value = line.split('\t')
#         data_list.append((key, value))

# df = pd.DataFrame(data_list, columns=['word', 'phon'])
# print(df)

# file_path = 'phoneticDictionary.csv'
# df.to_csv(file_path, index=False)
        
df = pd.read_csv('../vectorizer/hard_words/phonetic_dictionary.csv')
df = pd.DataFrame(list(zip(df['word'], df['phon'])), columns=['word', 'ipa'])

def ipa_word(word):
    this_word = ''
    ipa_word = list(df['word'])
    ipa_notation = list(df['ipa'])
    ipa_dict = dict(zip(ipa_word, ipa_notation))
    if word in ipa_dict.keys():
        this_word = ipa_dict[word].replace("ˈ", "")
        this_word = this_word.replace("ˌ", "")
    return this_word


def check_assimilated_row(row):
    assimilated = 0
    retWords = []
    for word in row:
        if len(magic_e_result(word)) > 1:
            if re.search('^ill', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^imm[aeiou]', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^imp', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^irr[aeiou]', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^suff', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^supp', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^succ', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^surr', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^coll', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^corr', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^att', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^aff', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^agg', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^all', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^ann', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^app', word):
                if not re.search('apples', word):
                    assimilated += 1
                    if word not in retWords:
                        retWords.append(word)
            if re.search('^ass', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^arr', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^diff', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^eff', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^opp', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^off', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)
            if re.search('^occ', word):
                assimilated += 1
                if word not in retWords:
                    retWords.append(word)

    return retWords

def check_adv_suffix_word(row, pos):

    adjs_nouns = ['JJR', 'JJS', 'JJ', 'NN', 'NNP', 'NNS']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP']

    adj_n_suffix = ['ɛɹi$', 'ɔɹi$', 'ənsi$', 'əns$', 'ʒən', 'ʒən', 'ʃən', 'əbəɫ$', 'əbɫi$']
    v_suffix = ['aɪz', 'ɪfaɪ', 'əfaɪ']

    ipa = []
    ret_dict = {}
    for word in row:
        ipa.append(ipa_word(word))
        ret_dict[ipa_word(word)] = word
    i = 0
    retWords = []
    for word in ipa:
        if len(word) > 0:
            if magic_e(row[i]) > 1:
                if pos[i] in adjs_nouns:
                    for suf in adj_n_suffix:
                        if re.search(suf, word):
                            if ret_dict[word] not in retWords:
                                retWords.append(ret_dict[word])
                if pos[i] in verbs:
                    for suf in v_suffix:
                        if re.search(suf, word):
                            if ret_dict[word] not in retWords:
                                retWords.append(ret_dict[word])

        i += 1

    return retWords

COMPOUND_WORDS = ['aircraft', 'airline', 'airmail', 'airplane', 'airport', 'airtight', 'anybody', 'anymore', 'anyone',
                 'anyplace', 'anything', 'anywhere', 'anyhow', 'backboard', 'backbone', 'backfire', 'background',
                 'backpack', 'backward', 'backyard', 'bareback', 'feedback', 'flashback', 'hatchback', 'paperback',
                 'piggyback', 'bathrobe', 'bathroom', 'bathtub', 'birdbath', 'bedrock', 'bedroom', 'bedside',
                 'bedspread', 'bedtime', 'flatbed', 'hotbed', 'sickbed', 'waterbed', 'birthday', 'birthmark',
                 'birthplace', 'birthstone', 'childbirth', 'blackberry', 'blackmail', 'blacksmith', 'blacktop',
                 'bookcase', 'bookkeeper', 'bookmark', 'bookworm', 'checkbook', 'cookbook', 'scrapbook', 'textbook',
                 'buttercup', 'butterfly', 'buttermilk', 'butterscotch', 'doorbell', 'doorknob', 'doorman', 'doormat',
                 'doorstop', 'doorway', 'backdoor', 'outdoor', 'downcast', 'downhill', 'download', 'downpour',
                 'downright', 'downsize', 'downstairs', 'downstream', 'downtown', 'breakdown', 'countdown', 'sundown',
                 'touchdown', 'eyeball', 'eyebrow', 'eyeglasses', 'eyelash', 'eyelid', 'eyesight', 'eyewitness',
                 'shuteye', 'firearm', 'firecracker', 'firefighter', 'firefly', 'firehouse', 'fireman', 'fireplace',
                 'fireproof', 'fireside', 'firewood', 'fireworks', 'backfire', 'bonfire', 'campfire', 'football',
                 'foothill', 'foothold', 'footlights', 'footnote', 'footprint', 'footstep', 'footstool', 'barefoot',
                 'tenderfoot', 'grandchildren', 'granddaughter', 'grandfather', 'grandmother', 'grandparent',
                 'grandson', 'haircut', 'hairdo', 'hairdresser', 'hairpin', 'hairstyle', 'handbag', 'handball',
                 'handbook', 'handcuffs', 'handmade', 'handout', 'handshake', 'handspring', 'handstand',
                 'handwriting', 'backhand', 'firsthand', 'secondhand', 'underhand', 'headache', 'headband',
                 'headdress', 'headfirst', 'headlight', 'headline', 'headlong', 'headmaster', 'headphones',
                 'headquarters', 'headstart', 'headstrong', 'headway', 'airhead', 'blockhead', 'figurehead',
                 'homeland', 'homemade', 'homemaker', 'homeroom', 'homesick', 'homespun', 'homestead', 'homework',
                 'horseback', 'horsefly', 'horseman', 'horseplay', 'horsepower', 'horseshoe', 'racehorse', 'sawhorse',
                 'houseboat', 'housefly', 'housewife', 'housework', 'housetop', 'birdhouse', 'clubhouse', 'doghouse',
                 'greenhouse', 'townhouse', 'landfill', 'landlady', 'landlord', 'landmark', 'landscape', 'landslide',
                 'dreamland', 'farmland', 'highland', 'wasteland', 'wonderland', 'lifeboat', 'lifeguard',
                 'lifejacket', 'lifelike', 'lifelong', 'lifestyle', 'lifetime', 'nightlife', 'wildlife', 'lighthouse',
                 'lightweight', 'daylight', 'flashlight', 'headlight', 'moonlight', 'spotlight', 'sunlight',
                 'mailman', 'snowman', 'gentleman', 'handyman', 'policeman', 'salesman', 'nightfall', 'nightgown',
                 'nightmare', 'nightlight', 'nighttime', 'overnight', 'outbreak', 'outcast', 'outcome', 'outcry',
                 'outdated', 'outdo', 'outdoors', 'outfield', 'outfit', 'outgrow', 'outlaw', 'outline', 'outlook',
                 'outnumber', 'outpost', 'outrage', 'outright', 'outside', 'outsmart', 'outwit', 'blowout',
                 'carryout', 'cookout', 'handout', 'hideout', 'workout', 'lookout', 'overall', 'overboard',
                 'overcast', 'overcome', 'overflow', 'overhead', 'overlook', 'overview', 'playground', 'playhouse',
                 'playmate', 'playpen', 'playroom', 'playwright', 'rainbow', 'raincoat', 'raindrop', 'rainfall',
                 'rainstorm', 'roadblock', 'roadway', 'roadwork', 'railroad', 'sandbag', 'sandbar', 'sandbox',
                 'sandpaper', 'sandpiper', 'sandstone', 'seacoast', 'seafood', 'seagull', 'seaman', 'seaport',
                 'seasick', 'seashore', 'seaside', 'seaweed', 'snowball', 'snowflake', 'snowplow', 'snowshoe',
                 'snowstorm', 'somebody', 'someone', 'someday', 'somehow', 'somewhere', 'something', 'sometime',
                 'underline', 'undergo', 'underground', 'undermine', 'underwater', 'watercolor', 'waterfall',
                 'watermelon', 'waterproof', 'saltwater', 'windfall', 'windmill', 'windpipe', 'windshield',
                 'windswept', 'downwind', 'headwind']

def easy_check_compound(row):

    compounds = []
    for word in row:
        if word in COMPOUND_WORDS:
            if word not in compounds:
                compounds.append(word)

    return compounds

def check_adv_inflectional_verb_row(text_POS):

    verb_tags = ['VBD', 'VBG', 'VBN', 'VBZ']
    inflectional = []

    for item in text_POS:
        if item[1] in verb_tags:
            if re.search('ing$', item[0]):
                inflectional.append(item[0])
            if re.search('ed$', item[0]):
                inflectional.append(item[0])

    return inflectional

def check_adv_inflectional_adj_row(text_POS):

    adj_tags = ['JJR', 'JJS', 'RBR', 'RBS', 'JJ', 'RB']
    inflectional = []

    for item in text_POS:
        if len(item[0]) > 4:
            if item[1] in adj_tags:
                if item[1] == 'JJ':
                    if re.search('ful$', item[0]):
                        if item[0] not in inflectional:
                            inflectional.append(item[0])
                    if re.search('ness$', item[0]):
                        if item[0] not in inflectional:
                            inflectional.append(item[0])
                    if re.search('less$', item[0]):
                        if item[0] not in inflectional:
                            inflectional.append(item[0])
                    if re.search('ily$', item[0]):
                        if item[0] not in inflectional:
                            inflectional.append(item[0])
                else:
                    if item[0] not in inflectional:
                        inflectional.append(item[0])

    return inflectional

def calculate(snippet):

    text = re.sub(r'[^\w\s]', '', snippet)
    text = text.lower().split()
    num_words = len(text)
    pos = get_POS(text)
    text_POS = list(zip(text, pos))

    assim = check_assimilated_row(text)
    adv_suf = check_adv_suffix_word(text, pos)
    compound_words = easy_check_compound(text)
    inf_verb = check_adv_inflectional_verb_row(text_POS)
    inf_adj = check_adv_inflectional_adj_row(text_POS)

    all_words = []

    for word in assim:
        if word not in all_words:
            all_words.append(word)
    for word in adv_suf:
        if word not in all_words:
            all_words.append(word)
    for word in compound_words:
        if word not in all_words:
            all_words.append(word)
    for word in inf_verb:
        if word not in all_words:
            all_words.append(word)
    for word in inf_adj:
        if word not in all_words:
            all_words.append(word)

    return all_words, num_words


def get_hard_words(snippet):
    
    hard_words, num_words_snippet = calculate(snippet)
    num_hard_words = len(hard_words)
    if num_words_snippet == 0:
        return hard_words, num_hard_words, 0
    proportion_hard_words = num_hard_words / num_words_snippet
    return hard_words, num_hard_words, proportion_hard_words


def get_hard_words_from_list(snippets):
    proportions_of_hard_words = []
    nums_hard_words = []
    
    for snippet in snippets:
        hard_words, num_hard_words, proportion_hard_words = get_hard_words(snippet)
        proportions_of_hard_words.append(proportion_hard_words)
        nums_hard_words.append(num_hard_words)
        
    return proportions_of_hard_words, nums_hard_words


def get_hard_words_from_file(file_path):
    proportions_of_hard_words = []
    nums_hard_words = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            hard_words, num_hard_words, proportion_hard_words = get_hard_words(line)
            proportions_of_hard_words.append(proportion_hard_words)
            nums_hard_words.append(num_hard_words)
            
    return proportions_of_hard_words, nums_hard_words


# prop, nums = get_hard_words_from_list(["\n\n[Produced by T-Minus]\n\n[Intro]\nPour up (Drank), head shot (Drank)\nSit down (Drank), stand up (Drank)\nPass out (Drank), wake up (Drank)\nFaded (Drank), faded (Drank)\n\n[Verse 1]\nNow I done grew up 'round some people livin' their life in bottles\nGranddaddy had the golden flask\nBackstroke every day in Chicago\nSome people like the way it feels\nSome people wanna kill their sorrows\nSome people wanna fit in with the popular, that was my problem\nI was in a dark room, loud tunes\nLookin' to make a vow soon\nThat I'ma get fucked up, fillin' up my cup I see the crowd mood\nChangin' by the minute and the record on repeat\nTook a sip, then another sip, then somebody said to me\n\n[Chorus]\nNigga, why you babysittin' only two or three shots?\nI'ma show you how to turn it up a notch\nFirst you get a swimming pool full of liquor, then you dive in it\nPool full of liquor, then you dive in it\nI wave a few bottles, then I watch 'em all flock\nAll the girls wanna play Baywatch\nI got a swimming pool full of liquor and they dive in it\nPool full of liquor, I'ma dive in it\n\n[Refrain]\nPour up (Drank), head shot (Drank)\nSit down (Drank), stand up (Drank)\nPass out (Drank), wake up (Drank)\nFaded (Drank), faded (Drank)\n\n[Verse 2]\nOkay, now open your mind up and listen me, Kendrick\nI am your conscience, if you do not hear me\nThen you will be history, Kendrick\nI know that you're nauseous right now\nAnd I'm hopin' to lead you to victory, Kendrick\nIf I take another one down\nI'ma drown in some poison, abusin' my limit\nI think that I'm feelin' the vibe, I see the love in her eyes\nI see the feelin', the freedom is granted\nAs soon as the damage of vodka arrived\nThis how you capitalize, this is parental advice\nThen apparently, I'm over-influenced by what you are doin'\nI thought I was doin' the most 'til someone said to me\n\n[Chorus]\nNigga, why you babysittin' only two or three shots?\nI'ma show you how to turn it up a notch\nFirst you get a swimming pool full of liquor, then you dive in it\nPool full of liquor, then you dive in it\nI wave a few bottles, then I watch 'em all flock\nAll the girls wanna play Baywatch\nI got a swimming pool full of liquor and they dive in it\nPool full of liquor, I'ma dive in it\n\n[Refrain]\nPour up (Drank), head shot (Drank)\nSit down (Drank), stand up (Drank)\nPass out (Drank), wake up (Drank)\nFaded (Drank), faded (Drank)\n\n[Bridge]\nI ride, you ride, bang\nOne chopper, 100 shots, bang\nHop out, do you bang?\nTwo chopper, 200 shots, bang\nI ride, you ride, bang\nOne chopper, 100 shots, bang\nHop out, do you bang?\nTwo chopper, 200 shots, bang\n\n[Chorus]\nNigga, why you babysittin' only two or three shots?\nI'ma show you how to turn it up a notch\nFirst you get a swimming pool full of liquor, then you dive in it\nPool full of liquor, then you dive in it\nI wave a few bottles, then I watch 'em all flock\nAll the girls wanna play Baywatch\nI got a swimming pool full of liquor and they dive in it\nPool full of liquor, I'ma dive in it\n\n[Refrain]\nPour up (Drank), head shot (Drank)\nSit down (Drank), stand up (Drank)\nPass out (Drank), wake up (Drank)\nFaded (Drank), faded (Drank)\n\n[Interlude: Sherane]\nAw man\u2026 where is she takin' me?\nWhere is she takin' me?\n\n[Verse 3]\nAll I have in life is my new appetite for failure\nAnd I got hunger pain that grow insane\nTell me, do that sound familiar?\nIf it do, then you're like me\nMakin' excuse that your relief\nIs in the bottom of the bottle and the greenest indo leaf\nAs the window open I release\nEverything that corrode inside of me\nI see you jokin', why you laugh?\nDon't you feel bad? I probably sleep\nAnd never ever wake up, never ever wake up, never ever wake up\nIn God I trust, but just when I thought I had enough\n\n[Outro]\nThey stomped the homie out over a bitch?\nK-dot, you good, blood?\nNow we can drop, ye we can drop you back off\nThat nigga's straight, man, that nigga ain't trippin'\nWe gon' do the same ol' shit\nI'ma pop a few shots, they gon' run, they gon' run opposite ways\nFall right in ****'s lap\nAnd he gon' tear they ass up, simple as that\nAnd I hope that bitch that set him up out there\nWe gon' pop that bitch too\nWait hold up, ayy, I see somebody\n[Car door opens and gunshots are fired]\nAha! Got them niggas, K-Dot, you good?\nL****, you good?\nYeah, blood, I'm good \u2013 Dave, you good?\nDave? Dave, say somethin' \u2013 Dave?\nThese bitch-ass niggas killed my brother!\n\n",
#                                        "\n\n[Produced by DJ Dahi]\n\n[Verse 1: Kendrick Lamar]\nUh, me and my niggas tryna get it, ya bish (ya bish)\nHit the house lick: tell me, is you wit' it, ya bish? (ya bish)\nHome invasion was persuasive (was persuasive)\nFrom nine to five I know it's vacant, ya bish (ya bish)\nDreams of livin' life like rappers do (like rappers do)\nBack when condom wrappers wasn't cool (they wasn't cool)\nI fucked Sherane and went to tell my bros (tell my bros)\nThen Usher Raymond \"Let It Burn\" came on (\"Let Burn\" came on)\nHot sauce all in our Top Ramen, ya bish (ya bish)\nPark the car, then we start rhymin', ya bish (ya bish)\nThe only thing we had to free our mind (free our mind)\nThen freeze that verse when we see dollar signs (see dollar signs)\nYou lookin' like a easy come-up, ya bish (ya bish)\nA silver spoon I know you come from, ya bish (ya bish)\nAnd that's a lifestyle that we never knew (we never knew)\nGo at a reverend for the revenue\n\n[Hook: Kendrick Lamar]\nIt go Halle Berry or hallelujah\nPick your poison, tell me what you doin'\nEverybody gon' respect the shooter\nBut the one in front of the gun lives forever\n(The one in front of the gun, forever)\nAnd I been hustlin' all day\nThis-a-way, that-a-way\nThrough canals and alleyways, just to say\nMoney trees is the perfect place for shade\nAnd that's just how I feel, nah, nah\nA dollar might just fuck your main bitch\nThat's just how I feel, nah\nA dollar might say fuck them niggas that you came with\nThat's just how I feel, nah, nah\nA dollar might just make that lane switch\nThat's just how I feel, nah\nA dollar might turn to a million and we all rich\nThat's just how I feel\n\n[Verse 2: Kendrick Lamar]\nDreams of livin' life like rappers do (like rappers do)\nBump that new E-40 after school (way after school)\nYou know, \u201cBig Ballin' With My Homies\u201d (my homies)\nEarl Stevens had us thinkin' rational (thinkin' rational)\nBack to reality, we poor, ya bish (ya bish)\nAnother casualty at war, ya bish (ya bish)\nTwo bullets in my Uncle Tony head (my Tony head)\nHe said one day I'll be on tour, ya bish (ya bish)\nThat Louis Burgers never be the same (won't be the same)\nA Louis belt will never ease that pain (won't ease that pain)\nBut I'ma purchase when that day is jerkin' (that day is jerkin')\nPull off at Church's, with Pirellis skirtin' (Pirellis skirtin')\nGang signs out the window, ya bish (ya bish)\nHopin' all of 'em offend you, ya bish (ya bish)\nThey say your hood is a pot of gold (a pot of gold)\nAnd we gon' crash it when nobody's home\n\n[Hook: Kendrick Lamar]\nIt go Halle Berry or hallelujah\nPick your poison, tell me what you doin'\nEverybody gon' respect the shooter\nBut the one in front of the gun lives forever\n(The one in front of the gun, forever)\nAnd I been hustlin' all day\nThis-a-way, that-a-way\nThrough canals and alleyways, just to say\nMoney trees is the perfect place for shade\nAnd that's just how I feel, nah, nah\nA dollar might just fuck your main bitch\nThat's just how I feel, nah\nA dollar might say fuck them niggas that you came with\nThat's just how I feel, nah, nah\nA dollar might just make that lane switch\nThat's just how I feel, nah\nA dollar might turn to a million and we all rich\nThat's just how I feel\n\n[Bridge: Anna Wise]\nBe the last one out to get this dough? No way!\nLove one of you bucket-headed hoes? No way!\nHit the streets, then we break the code? No way!\nHit the brakes when they on patrol? No way!\nBe the last one out to get this dough? No way!\nLove one of you bucket-headed hoes? No way!\nHit the streets, then we break the code? No way!\nHit the brakes when they on patrol? No way!\n\n[Verse 3: Jay Rock]\nImagine Rock up in them projects\nWhere them niggas pick your pockets\nSanta Claus don't miss them stockings\nLiquors spillin', pistols poppin'\nBakin' soda YOLA whippin'\nAin't no turkey on Thanksgivin'\nMy homeboy just dome'd a nigga\nI just hope the Lord forgive him\nPots with cocaine residue\nEvery day I'm hustlin'\nWhat else is a thug to do\nWhen you eatin' cheese from the government?\nGotta provide for my daughter n'em\nGet the fuck up out my way, bish\nGot that drum and I got them bands\nJust like a parade, bish\nDrop that work up in the bushes\nHope them boys don't see my stash\nIf they do, tell the truth\nThis the last time you might see my ass\nFrom the gardens where the grass ain't cut\nThem serpents lurkin', Blood\nBitches sellin' pussy, niggas sellin' drugs\nBut it's all good\nBroken promises, steal your watch\nAnd tell you what time it is\nTake your J's and tell you to kick it where a FootLocker is\nIn the streets with a heater under my Dungarees\nDreams of me gettin' shaded under a money tree\n\n[Hook: Kendrick Lamar]\nIt go Halle Berry or hallelujah\nPick your poison, tell me what you doin'\nEverybody gon' respect the shooter\nBut the one in front of the gun lives forever\n(The one in front of the gun, forever)\nAnd I been hustlin' all day\nThis-a-way, that-a-way\nThrough canals and alleyways, just to say\nMoney trees is the perfect place for shade\nAnd that's just how I feel\n\n[Outro]\nK\u2019s Mom: Kendrick, just bring my car back, man. I called in for another appointment. I figured you weren\u2019t gonna be back here on time anyways. Look, shit, shit, I just wanna get out the house, man. This man is on one, he feelin' good as a motherfucker. Shit, I\u2019m tryna get my thing goin', too. Just bring my car back. Shit, he faded. He feelin' good. Look, listen to him!\nK\u2019s Dad: Girl, girl, I want your body, I want your body, 'cause of that big ol\u2019 fat ass. Girl, girl, I want your body, I want your body, 'cause of that big ol\u2019 fat ass\nK\u2019s Mom: See, he high as hell. Shit, and he ain\u2019t even trippin' off them damn dominoes anymore. Just bring the car back!\nK\u2019s Dad: Did somebody say dominoes?\n\n",
#                                        "\n\n[Intro: B\u0113kon & Kid Capri]\nAmerica, God bless you if it's good to you\nAmerica, please take my hand\nCan you help me underst\u2014\nNew Kung Fu Kenny!\n\n[Verse 1: Kendrick Lamar]\nThrow a steak off the ark to a pool full of sharks, he'll take it\nLeave him in the wilderness with a sworn nemesis, he'll make it\nTake the gratitude from him, I bet he'll show you somethin', whoa\nI'll chip a nigga lil' bit of nothin', I'll chip a nigga lil' bit of nothin'\nI'll chip a nigga lil' bit of nothin', I'll chip a nigga, then throw the blower in his lap\nWalk myself to the court like, \"Bitch, I did that!,\" X-rated\nJohnny don't wanna go to school no mo', no mo'\nJohnny said books ain't cool no mo' (No mo')\nJohnny wanna be a rapper like his big cousin\nJohnny caught a body yesterday out hustlin'\nGod bless America, you know we all love him\n\n[Verse 2: Kendrick Lamar]\nYesterday I got a call like from my dog like 101\nSaid they killed his only son because of insufficient funds\nHe was sobbin', he was mobbin', way belligerent and drunk\nTalkin' out his head, philosophin' on what the Lord had done\nHe said: \"K-Dot, can you pray for me?\nIt been a fucked up day for me\nI know that you anointed, show me how to overcome.\"\nHe was lookin' for some closure\nHopin' I could bring him closer\nTo the spiritual, my spirit do know better, but I told him\n\"I can't sugarcoat the answer for you, this is how I feel:\nIf somebody kill my son, that mean somebody gettin' killed.\"\nTell me what you do for love, loyalty, and passion of\nAll the memories collected, moments you could never touch\nI'll wait in front a nigga's spot and watch him hit his block\nI'll catch a nigga leavin' service if that's all I got\nI'll chip a nigga, then throw the blower in his lap\nWalk myself to the court like, \"Bitch, I did that!\"\nAin't no Black Power when your baby killed by a coward\nI can't even keep the peace, don't you fuck with one of ours\nIt be murder in the street, it be bodies in the hour\nGhetto bird be on the street, paramedics on the dial\nLet somebody touch my momma\nTouch my sister, touch my woman\nTouch my daddy, touch my niece\nTouch my nephew, touch my brother\nYou should chip a nigga, then throw the blower in his lap\nMatter fact, I'm 'bout to speak at this convention\nCall you back\u2014\n\n[Break: Kendrick Lamar]\nAlright, kids, we're gonna talk about gun control\n(Pray for me) Damn!\n\n[Chorus: Bono]\nIt's not a place\nThis country is to me a sound\nOf drum and bass\nYou close your eyes to look around\n\n[Verse 3: Kendrick Lamar]\nHail Mary, Jesus and Joseph\nThe great American flag is wrapped in drag with explosives\nCompulsive disorder, sons and daughters\nBarricaded blocks and borders\nLook what you taught us!\nIt's murder on my street, your street, back streets\nWall Street, corporate offices\nBanks, employees, and bosses with\nHomicidal thoughts; Donald Trump's in office\nWe lost Barack and promised to never doubt him again\nBut is America honest, or do we bask in sin?\nPass the gin, I mix it with American blood\nThen bash him in, you Crippin' or you married to Blood?\nI'll ask again\u2014oops, accident\nIt's nasty when you set us up\nThen roll the dice, then bet us up\nYou overnight the big rifles, then tell Fox to be scared of us\nGang members or terrorists, et cetera, et cetera\nAmerica's reflections of me, that's what a mirror does\n\n[Chorus: Bono]\nIt's not a place\nThis country is to me a sound\nOf drum and bass\nYou close your eyes to look ar\u2014\n\n",
#                                        "\n\n[Chorus: 2 Chainz, Drake & Both (A$AP Rocky)]\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nIf findin' somebody real is your fuckin' problem (Yeah)\nBring your girls to the crib, maybe we can solve it, ayy\n\n[Verse 1: A$AP Rocky]\nHold up, bitches simmer down (Uh)\nTakin' hella long, bitch, give it to me now (Uh)\nMake that thing pop like a semi or a 9\nOoh, baby like it raw with the shimmy shimmy ya, huh?\nA$AP (Yeah,) get like me (Uh)\nNever met a motherfucker fresh like me (Yeah)\nAll these motherfuckers wanna dress like me (Uh)\nBut the chrome to your dome make you sweat like Keith\n\u2018Cause I'm the nigga, the nigga nigga, like how you figure? (Yeah)\nGettin' figures and fuckin' bitches, she rollin' Swishers\nBrought her bitches, I brought my niggas (Uh)\nThey gettin' bent up off the liquor (Uh)\nShe love my licorice, I let her lick it (Alright)\nThey say money make a nigga act niggerish (Uh)\nBut least a nigga nigga rich\nI be fuckin' broads like I be fuckin' bored\nTurn a dyke bitch out, have her fuckin' boys; beast\n\n[Chorus: 2 Chainz, Drake & Both]\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nIf findin' somebody real is your fuckin' problem\nBring your girls to the crib, maybe we can solve it, ayy\n\n[Verse 2: Drake]\nOoh, I know you love it when this beat is on\nMake you think about all of the niggas you been leadin' on\nMake me think about all of the rappers I've been feedin' on\nGot a feelin' that's the same dudes that we speakin' on, oh word?\nAin't heard my album? Who you sleepin' on?\nYou should print the lyrics out and have a fuckin' read-along\nAin't a fuckin' sing-along 'less you brought the weed along\nThen ju\u2026 okay, I got it\nThen just drop down and get your eagle on\nOr we can stare up at the stars and put the Beatles on\nAll that shit you talkin' 'bout is not up for discussion\nI will pay to make it bigger, I don't pay for no reduction\nIf it's comin' from a nigga I don't know, then I don't trust it\nIf you comin' for my head, then motherfucker get to bustin'\nYes, Lord, I don't really say this often\nBut this long-dick nigga ain't for the long talkin'; I'm beast\n\n[Chorus: 2 Chainz, Drake & Both]\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nIf findin' somebody real is your fuckin' problem\nBring your girls to the crib, maybe we can solve it\n\n[Verse 3: Kendrick Lamar]\nUh, yeah, ho, this the finale\nMy pep talk turn into a pep rally\nSay she from the hood, but she live inside in the valley, now\nVaca'd in Atlanta, then she goin' back to Cali, mmm\nGot your girl on my line, world on my line\nThe irony, I fuck 'em at the same damn time\nShe eyein' me like a nigga don't exist\nGirl, I know you want this dick\nGirl, I'm Kendrick Lamar (Uh)\nA.K.A. Benz is to me just a car (Uh)\nThat mean your friends-es need be up to par\nSee, my standards are pampered by threesomes tomorrow\nMmm, kill 'em all, dead bodies in the hallway\nDon't get involved, listen what the crystal ball say\nHalle Berry, hallelujah\nHolla back, I'll do ya; beast\n\n[Chorus: 2 Chainz, Drake & Both]\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nI love bad bitches, that's my fuckin' problem\nAnd yeah, I like to fuck, I got a fuckin' problem\nIf findin' somebody real is your fuckin' problem\nBring your girls to the crib, maybe we can solve it, ayy\n\n[Produced by 40 & Drake]\n\n"])        


# print(prop, nums)