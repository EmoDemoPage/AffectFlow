import json
import os

from data_gen.tts.base_preprocess import BasePreprocessor

class DailyTalkPreprocess(BasePreprocessor):
    def meta_data(self):
        txt_path = os.path.join(self.raw_data_dir, 'metadata.txt')
        
        emotion_dict = {"disgust": 0, "none": 1, "surprise": 2, "happiness": 3, "fear": 4, "sadness": 5, "anger": 6}
        intensity_dict = {"1": 0, "2": 1, "3": 2}

        for l in open(txt_path).readlines():
            item_name, spk_name, phone, txt, emotion, intensity = l.strip().split("|")
            
            utterance_idx = item_name.split('_')[0]
            dialog_idx = item_name.split('_')[-1].strip('d')
            wav_fn = os.path.join(self.raw_data_dir, 'data', str(dialog_idx), f"{item_name}.wav")
            # print(wav_fn)
            emotion = emotion_dict[emotion]
            intensity = intensity_dict[intensity]

            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'emotion': emotion, 'intensity': intensity, 'spk_name': spk_name}
