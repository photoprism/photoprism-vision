
import yaml

class LabelRulesProcessor:
    def __init__(self, rules_path):
        self.rules = {}
        
        self._load_rules(rules_path)

    def _load_rules(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)

    def transform_label(self, label_key, pred_score):
        outputs = {
            'state':'fail',
            'label':None,
            'categories':None,
            'conf':None,
        }
        if(label_key not in self.rules):
            return outputs
        
        label_property = self.rules[label_key]
        if 'see' in label_property:
            label_property = self.rules[label_property['see']]

        # get label
        if('label' in label_property):
            new_label = label_property['label']
        elif('categories' in label_property):
            new_label = label_property['categories'][0]
        else:
            outputs['state'] = 'fail'
            return outputs
        # get categories
        if('categories' in label_property):
            categories = label_property['categories']
        else:
            categories = None
        
        # get threshold
        if('threshold' in label_property):
            threshold = label_property['threshold']
        else:
            threshold = 0.5
        
        # condition
        if(pred_score>threshold):
            outputs['state'] = 'sucess'
            outputs['label'] = new_label
            outputs['categories'] = categories
            outputs['conf'] = pred_score
            return outputs
        else:
            outputs['state'] = 'fail'
            return outputs