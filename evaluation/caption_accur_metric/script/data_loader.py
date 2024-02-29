import json
import jsonlines
import os
from nltk.tokenize import RegexpTokenizer

class BLEUDataLoader:
    def __init__(self,
                 ref_path="../captions/ref/gpt4v_llava_10k_test.json", ref_repeat=4,
                 hypo_dir="../captions/hypo", split=True) -> None:
        self.ref_path = ref_path
        self.ref_repeat = ref_repeat
        self.hypo_dir = hypo_dir
        self.split = split
        self.ref = self.set_ref()
        self.hypo_dict = self.set_hypo_dict()
        
    
    def corpus_process(self, paragraph):
        if self.split == False:
            return paragraph
        tokenizer = RegexpTokenizer(r'\w+')
        pure_words = tokenizer.tokenize(paragraph)
        return pure_words

    def set_ref(self):
        # [[ref_a1],]
        ref = []
        with open(self.ref_path, 'r') as f:
            content = json.load(f)
        for i in range(int(len(content)/self.ref_repeat)):
            ref.append([self.corpus_process(content[self.ref_repeat*i]["conversations"][1]["value"])])
        return ref
    
    def gen_hypo(self, hypo_path="../captions/hypo/output-caption-gpt4v-hypo-0.01.jsonl"):
        # [hypo_a,]
        hypo = []
        with open(hypo_path, 'r') as f:
            for chat in jsonlines.Reader(f):
                hypo.append(self.corpus_process(chat['outputs']))
        return hypo
    
    def set_hypo_dict(self):
        # {filename_attr: [hypo_a,]}
        hypo_dict = {}
        for file_name in os.listdir(self.hypo_dir):
            hypo_path = os.path.join(self.hypo_dir, file_name)
            filename_attr = file_name[21:-6]
            hypo = self.gen_hypo(hypo_path)
            hypo_dict[filename_attr] = hypo
        return hypo_dict

class CIDErDataLoader:
    def __init__(self,
                 ref_path="../captions/ref/gpt4v_llava_10k_test.json", ref_repeat=4,
                 hypo_dir="../captions/hypo", split=True) -> None:
        self.ref_path = ref_path
        self.ref_repeat = ref_repeat
        self.hypo_dir = hypo_dir
        self.split = split
        self.img = self.set_img()
        self.ref = self.set_ref()
        self.hypo_dict = self.set_hypo_dict()
        
    
    def corpus_process(self, paragraph):
        if self.split == False:
            return paragraph
        tokenizer = RegexpTokenizer(r'\w+')
        pure_words = tokenizer.tokenize(paragraph)
        return pure_words

    def set_img(self):
        img = []
        with open(self.ref_path, 'r') as f:
            content = json.load(f)
        for i in range(int(len(content)/self.ref_repeat)):
            img.append(int(content[self.ref_repeat*i]["image"].split('.')[0]))
        return img

    def set_ref(self):
        # {<image>: [<tokenized reference sentence>]}
        ref = {}
        with open(self.ref_path, 'r') as f:
            content = json.load(f)
        for i in range(int(len(content)/self.ref_repeat)):
            ref[self.img[len(ref)]] = [self.corpus_process(content[self.ref_repeat*i]["conversations"][1]["value"])]
        return ref
    
    def gen_hypo(self, hypo_path="../captions/hypo/output-caption-gpt4v-hypo-0.01.jsonl"):
        # {<image>: [<tokenized hypothesis sentence>]}
        hypo = {}
        with open(hypo_path, 'r') as f:
            for chat in jsonlines.Reader(f):
                hypo[self.img[len(hypo)]] = [self.corpus_process(chat['outputs'])]
        return hypo
    
    def set_hypo_dict(self):
        # {filename_attr: {<image>: <tokenized hypothesis sentence>}}
        hypo_dict = {}
        for file_name in os.listdir(self.hypo_dir):
            hypo_path = os.path.join(self.hypo_dir, file_name)
            filename_attr = file_name[21:-6]
            hypo = self.gen_hypo(hypo_path)
            hypo_dict[filename_attr] = hypo
        return hypo_dict
 
class CIDErDDataLoader:
    def __init__(self,
                 ref_path="../captions/ref/gpt4v_llava_10k_test.json", ref_repeat=4,
                 hypo_dir="../captions/hypo", split=True) -> None:
        self.ref_path = ref_path
        self.ref_repeat = ref_repeat
        self.hypo_dir = hypo_dir
        self.split = split
        self.img = self.set_img()
        self.ref = self.set_ref()
        self.hypo_dict = self.set_hypo_dict()
        
    
    def corpus_process(self, paragraph):
        if self.split == False:
            return paragraph
        tokenizer = RegexpTokenizer(r'\w+')
        pure_words = tokenizer.tokenize(paragraph)
        return pure_words

    def set_img(self):
        img = []
        with open(self.ref_path, 'r') as f:
            content = json.load(f)
        for i in range(int(len(content)/self.ref_repeat)):
            img.append(int(content[self.ref_repeat*i]["image"].split('.')[0]))
        return img

    def set_ref(self):
        # {<image>: <tokenized reference sentence>}
        ref = {}
        with open(self.ref_path, 'r') as f:
            content = json.load(f)
        for i in range(int(len(content)/self.ref_repeat)):
            ref[self.img[len(ref)]] = self.corpus_process(content[self.ref_repeat*i]["conversations"][1]["value"])
        return ref
    
    def gen_hypo(self, hypo_path="../captions/hypo/output-caption-gpt4v-hypo-0.01.jsonl"):
        # {"image_id": <image>, "caption": <tokenized hypothesis sentence>}
        hypo = []
        with open(hypo_path, 'r') as f:
            for chat in jsonlines.Reader(f):
                res = {}
                res["image_id"] = self.img[len(hypo)]
                res["caption"] = self.corpus_process(chat['outputs'])
                hypo.append(res)
        return hypo
    
    def set_hypo_dict(self):
        # {filename_attr: {"image_id": <image>, "caption": <tokenized hypothesis sentence>}}
        hypo_dict = {}
        for file_name in os.listdir(self.hypo_dir):
            hypo_path = os.path.join(self.hypo_dir, file_name)
            filename_attr = file_name[21:-6]
            hypo = self.gen_hypo(hypo_path)
            hypo_dict[filename_attr] = hypo
        return hypo_dict
