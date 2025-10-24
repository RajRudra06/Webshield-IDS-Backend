from tldextract import extract
from difflib import SequenceMatcher
from app.constants.legitimate_brand_domains import * 

def generate_all_typosquatting_patterns(brand):
  
    patterns = set()
    
    substitutions = {
        'o': ['0'],
        'i': ['1', '!', '|'],
        'l': ['1', '|'],
        'e': ['3'],
        'a': ['@', '4'],
        's': ['$', '5'],
        'g': ['9'],
        't': ['7'],
        'b': ['8']
    }
    
    for i, char in enumerate(brand):
        if char in substitutions:
            for replacement in substitutions[char]:
                variant = brand[:i] + replacement + brand[i+1:]
                patterns.add(variant)
    
    for i, char1 in enumerate(brand):
        if char1 in substitutions:
            for j, char2 in enumerate(brand[i+1:], start=i+1):
                if char2 in substitutions:
                    for rep1 in substitutions[char1]:
                        for rep2 in substitutions[char2]:
                            variant = list(brand)
                            variant[i] = rep1
                            variant[j] = rep2
                            patterns.add(''.join(variant))
    
    if len(brand) > 4:  
        for i in range(len(brand)):
            variant = brand[:i] + brand[i+1:]
            patterns.add(variant)
    
    for i in range(len(brand)):
        variant = brand[:i] + brand[i] + brand[i:]
        patterns.add(variant)
    
    for i in range(len(brand) - 1):
        chars = list(brand)
        chars[i], chars[i+1] = chars[i+1], chars[i]
        patterns.add(''.join(chars))
    
    return patterns

def check_typosquatting_heuristic(domain_name):
   
    domain_lower = domain_name.lower()
    
    for legit in LEGITIMATE_BRAND_DOMAINS:
        if domain_name == legit.split('.')[0]:
            return False, None, 0.0
    
    for brand in BRAND_KEYWORDS:
        if domain_lower == brand:
            continue
        
        patterns = generate_all_typosquatting_patterns(brand)
        
        if domain_lower in patterns:
            return True, brand, 0.95  
        
        for pattern in patterns:
            if pattern in domain_lower:
                if not any(legit.startswith(domain_lower) for legit in LEGITIMATE_BRAND_DOMAINS):
                    return True, brand, 0.90
        
        similarity = SequenceMatcher(None, domain_lower, brand).ratio()
        if 0.75 <= similarity < 1.0:
            if len(domain_lower) <= len(brand) + 2:
                return True, brand, similarity
    
    return False, None, 0.0

def check_homograph_attack(domain_name):
 
    homoglyphs = {
        'a': ['а', '@', '4'],  
        'o': ['о', '0'],      
        'e': ['е', '3'],       
        'i': ['і', '1', '!', '|'],
        'l': ['1', '|', 'I'],
        'c': ['с'],            
        's': ['$', '5'],
        'b': ['8'],
        'g': ['9'],
        't': ['7'],
    }
    
    for char in domain_name.lower():
        if not char.isalnum():
            continue
        for original, replacements in homoglyphs.items():
            if char in replacements:
                test_domain = domain_name.lower().replace(char, original)
                if any(brand in test_domain for brand in BRAND_KEYWORDS):
                    return True, 0.85
    
    return False, 0.0

def apply_typosquatting_heuristic(url, model_prediction, model_probabilities):

    try:
        ext = extract(url)
        domain = ext.domain
        suffix = ext.suffix
        full_domain = f"{domain}.{suffix}".lower()
    except:
        return model_prediction, model_probabilities, "parsing_error"
    
    if full_domain in LEGITIMATE_BRAND_DOMAINS:
        return 'benign', {'benign': 0.999, 'phishing': 0.0005, 'malware': 0.0005, 'defacement': 0.0}, "whitelist_match"
    
    if model_prediction == 'phishing' and model_probabilities.get('phishing', 0) > 0.85:
        return model_prediction, model_probabilities, "model_confident"
    
    is_typo, matched_brand, typo_confidence = check_typosquatting_heuristic(domain)
    
    if is_typo and typo_confidence > 0.75:
        return 'phishing', {
            'benign': 0.05,
            'phishing': 0.92,
            'malware': 0.02,
            'defacement': 0.01
        }, f"typosquatting_{matched_brand}"
    
    is_homograph, homograph_conf = check_homograph_attack(domain)
    
    if is_homograph and homograph_conf > 0.75:
        return 'phishing', {
            'benign': 0.08,
            'phishing': 0.88,
            'malware': 0.03,
            'defacement': 0.01
        }, "homograph_attack"
    
    if model_prediction == 'benign':
        suspicious_score = 0
        reasons = []
        
        for brand in BRAND_KEYWORDS:
            if brand in domain.lower() and full_domain not in LEGITIMATE_BRAND_DOMAINS:
                suspicious_score += 0.3
                reasons.append(f"contains_{brand}")
        
        if any(c.isdigit() for c in domain) and any(c.isalpha() for c in domain):
            for brand in BRAND_KEYWORDS:
                domain_no_digits = ''.join(c for c in domain.lower() if not c.isdigit())
                if brand in domain_no_digits or SequenceMatcher(None, domain_no_digits, brand).ratio() > 0.75:
                    suspicious_score += 0.4
                    reasons.append("digits_in_brand")
                    break
        
        for brand in BRAND_KEYWORDS:
            if len(domain) <= len(brand) + 3:  
                similarity = SequenceMatcher(None, domain.lower(), brand).ratio()
                if 0.75 <= similarity < 1.0:
                    suspicious_score += 0.3
                    reasons.append(f"similar_to_{brand}")
        
        if suspicious_score >= 0.6:
            return 'phishing', {
                'benign': 0.15,
                'phishing': 0.80,
                'malware': 0.03,
                'defacement': 0.02
            }, f"heuristic_{'_'.join(reasons)}"
    
    return model_prediction, model_probabilities, "model_decision"
