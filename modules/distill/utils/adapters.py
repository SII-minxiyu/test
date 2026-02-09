import re

def get_adapter_model(config_dict, base_model):
    from peft import get_peft_model, get_peft_config
    patterns = config_dict['target_modules']
    target_modules = []
    for name, _ in base_model.named_modules():
        for pattern in patterns:
            if re.match(pattern, name):
                target_modules.append(name)
                
    config_dict['target_modules'] = target_modules 
    peft_config = get_peft_config(config_dict)
    
    return get_peft_model(base_model, peft_config)
