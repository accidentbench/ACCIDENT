import re
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import pandas as pd
import json
from reasoning.utils import match_to_class


class MolmoReasoner():
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
        ).to('cuda')

    
    def generate_text(self, img, prompt):
        """ Generate text for a given img and prompt """
        inputs = self.processor.process(
            images=[img],
            text=prompt
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
    
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )
        
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text


    def accident_spatial_reasoning(self, img, prompt=None):
        if prompt is None:
            prompt = (
                "The scene depicts a traffic accident with one or more cars colliding. "
                "Point to the car accident. "
            )
        generated_text = self.generate_text(img, prompt)
        point = self.parse_point(generated_text, img)
        return point, generated_text


    def accident_cause_reasoning(self, img, prompt):
        generated_text = self.generate_text(img, prompt)
        label = match_to_class(generated_text)
        return label, generated_text


    def accident_temporal_reasoning(self, imgs, prompt=None):
        if prompt is None:
            prompt = "Is there a traffic accident or collision? Yes or No answer only."
    
        preds_raw = {}
        for i, img in imgs:
            preds_raw[i] = self.generate_text(img, prompt)
    
        preds = {k: 'yes' in v.strip().lower() for k, v in preds_raw.items()}
        preds = pd.Series(preds)
        frame_ids = preds[preds].index
        if len(frame_ids) != 0:
            frame_id = frame_ids[0]
        else:
            frame_id = None

        return frame_id, preds_raw


    def parse_point(self, text, img):
        """
        Parses the first <point> tag and returns a (x, y) tuple of ints.
        Returns (None, None) if no match is found.
        """
        img_w, img_h = img.size
        pattern = r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"'
        match = re.search(pattern, text)
        if not match:
            return (None, None)

        x, y = match.groups()
        x, y = int(float(x)), int(float(y))
        x, y = int(float(x) * img_w / 100), int(float(y) * img_h / 100)
        return (x, y)
