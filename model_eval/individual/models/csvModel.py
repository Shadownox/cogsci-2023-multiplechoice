import ccobra
import numpy as np
import pandas as pd

class CSVModel(ccobra.CCobraModel):
    def __init__(self, name='csvModel', response_csv=None):
        super(CSVModel, self).__init__(name, ['syllogistic'], ['multiple-choice'])
        df = pd. read_csv(response_csv)
        
        self.responses = {}
        for syllog in ccobra.syllogistic.SYLLOGISMS:
            pred = df[df["Syllogism"] == syllog]["Prediction"].values[0].split(";")
            self.responses[syllog] = pred

    def predict(self, item, **kwargs):
            # Obtain task information
            syl = ccobra.syllogistic.Syllogism(item)
            enc_task = syl.encoded_task
            preds = self.responses[enc_task]
            
            return [syl.decode_response(x) for x in preds]