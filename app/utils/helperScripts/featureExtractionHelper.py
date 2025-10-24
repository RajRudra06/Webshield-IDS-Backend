import math 
from collections import Counter
import re
from difflib import SequenceMatcher
from app.constants.legitimate_brand_domains import * 

def shannon_entropy(s):
    if not isinstance(s, str) or len(s) == 0:
        return 0
    s = ''.join(c for c in s if 32 <= ord(c) <= 126)
    if len(s) == 0:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum([p * math.log2(p) for p in prob if p > 0])

def longest_repeated_char(s):
    if not s:
        return 0
    max_count = count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 1
    return max_count

def vowel_consonant_ratio(s):
    vowels = sum(1 for c in s.lower() if c in 'aeiou')
    consonants = sum(1 for c in s.lower() if c.isalpha() and c not in 'aeiou')
    return vowels / consonants if consonants > 0 else 0

def get_tld_category(tld):
    high_trust_tlds = ['com', 'org', 'net', 'edu', 'gov', 'mil', 'in', 'co.in', 'ac.in', 'gov.in']
    suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'cc', 'top', 'xyz', 'club', 'work', 'buzz', 'loan']

    if tld in high_trust_tlds:
        return 2  
    elif tld in suspicious_tlds:
        return 0  
    else:
        return 1  

def count_ngrams(s, n=2):
    if len(s) < n:
        return 0
    ngrams = [s[i:i + n] for i in range(len(s) - n + 1)]
    counter = Counter(ngrams)
    return len(counter)

def safe_max_len_list(values):
    try:
        return max(values) if values else 0
    except ValueError:
        return 0

def safe_max_match_length(pattern, text):
    try:
        matches = [len(m.group()) for m in re.finditer(pattern, text)]
        return max(matches) if matches else 0
    except Exception:
        return 0

def has_character_substitution(text):
  
    substitutions = {
        'o': '0', 'O': '0',
        'i': '1', 'I': '1', 'l': '1', 'L': '1',
        'e': '3', 'E': '3',
        'a': '@', 'A': '@',
        's': '$', 'S': '$',
        'g': '9', 'G': '9',
        't': '7', 'T': '7'
    }
    
    text_lower = text.lower()
    for original, replacement in substitutions.items():
        if replacement in text:
           
            for brand in BRAND_KEYWORDS:
                if original.lower() in brand:
                    pattern = brand.replace(original.lower(), replacement)
                    if pattern in text_lower:
                        return True
    return False

def check_advanced_typosquatting(domain, brand_list):
   
    domain_clean = domain.lower().replace('-', '').replace('_', '').replace('.', '')
    
    max_similarity = 0.0
    matched_brand = None
    is_typosquatting = False
    
    for brand in brand_list:
        if domain_clean == brand:
            return False, 1.0, brand
        
        if brand in domain_clean and len(domain_clean) > len(brand):
            is_typosquatting = True
            max_similarity = 0.85  
            matched_brand = brand
            continue
        
        similarity = SequenceMatcher(None, domain_clean, brand).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            matched_brand = brand
        
        if 0.7 <= similarity < 1.0:
            is_typosquatting = True
        
        if has_character_substitution(domain):
            for original, replacement in [('o', '0'), ('i', '1'), ('l', '1'), ('e', '3')]:
                pattern = brand.replace(original, replacement)
                if pattern in domain_clean:
                    is_typosquatting = True
                    max_similarity = 0.8
                    matched_brand = brand
                    break
    
    return is_typosquatting, max_similarity, matched_brand

