import ccobra
import numpy as np
import pandas as pd
import json

def to_uppercase(x):
    if x == "nvc":
        return "NVC"
    else:
        return x[0].upper() + x[1:]

class MFPModel(ccobra.CCobraModel):
    def __init__(self, name='MFA', response_csv=None):
        super(MFPModel, self).__init__(name, ['syllogistic'], ['multiple-choice'])
        
        with open('mfa.json', 'r') as f:
            self.patterns = json.load(f)

    def predict(self, item, **kwargs):
            # Obtain task information
            syl = ccobra.syllogistic.Syllogism(item)
            enc_task = syl.encoded_task
            preds = self.patterns[enc_task]
            
            pred = preds[0]
            if len(preds) > 1:
                pred = preds[np.random.randint(len(preds))]
            
            pred = [to_uppercase(x) for x in pred]
            
            return [syl.decode_response(x) for x in pred]