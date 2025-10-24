import re
from urllib.parse import urlparse
from tldextract import extract
from .featureExtractionHelper import *
from app.constants.legitimate_brand_domains import *

def extract_features_enhanced(url):
  
    features = {}

    try:
        if not isinstance(url, str) or len(url.strip()) == 0:
            raise ValueError("Invalid or empty URL string")

        url = ''.join(c for c in url if 32 <= ord(c) <= 126)

        parsed = urlparse(url)
        ext = extract(url)

        domain = ext.domain
        subdomain = ext.subdomain
        suffix = ext.suffix  # TLD
        path = parsed.path
        query = parsed.query
        netloc = parsed.netloc

        # basic/generic url features
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        features['num_special_chars'] = sum(url.count(c) for c in ['@', '?', '=', '%', '&', '!', '+', '$'])
        features['has_ip'] = int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', netloc)))
        features['num_subdomains'] = len(subdomain.split('.')) if subdomain else 0
        features['has_multiple_subdomains'] = int(features['num_subdomains'] >= 3)

        # domain related analysis
        features['domain_length'] = len(domain)
        features['host_entropy'] = shannon_entropy(domain)
        features['domain_entropy'] = shannon_entropy(domain)
        features['domain_has_digits'] = int(any(c.isdigit() for c in domain))
        features['domain_digit_ratio'] = sum(c.isdigit() for c in domain) / len(domain) if len(domain) > 0 else 0
        features['domain_vowel_ratio'] = vowel_consonant_ratio(domain)
        features['domain_bigram_diversity'] = count_ngrams(domain, 2) / len(domain) if len(domain) >= 2 else 0
        features['domain_trigram_diversity'] = count_ngrams(domain, 3) / len(domain) if len(domain) >= 3 else 0
        features['suspicious_prefix_suffix'] = int('-' in domain or domain.startswith('www-') or domain.startswith('m-'))
        features['num_suspicious_symbols'] = sum(domain.count(c) for c in ['@', '!', '*'])
        features['subdomain_length'] = len(subdomain) if subdomain else 0
        
        features['domain_is_dictionary_word'] = int(domain.lower() in BRAND_KEYWORDS)

        # tld extraction and analysis
        features['tld_length'] = len(suffix)
        features['tld_trust_category'] = get_tld_category(suffix.lower())
        features['is_suspicious_tld'] = int(suffix.lower() in ['tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'cc', 'top', 'xyz', 'club', 'work', 'info', 'biz', 'buzz', 'loan'])
        features['is_high_trust_tld'] = int(suffix.lower() in ['com', 'org', 'net', 'edu', 'gov', 'mil', 'in', 'co.in', 'ac.in', 'gov.in'])
        features['is_country_tld'] = int(len(suffix) == 2 and suffix.isalpha())

        # path/query analysis
        features['path_length'] = len(path)
        features['num_path_segments'] = len([p for p in path.split('/') if p])
        features['num_query_params'] = len(query.split('&')) if query else 0
        features['query_length'] = len(query)
        features['num_encoded_chars'] = url.count('%')
        features['num_fragments'] = url.count('#')
        features['path_entropy'] = shannon_entropy(path)
        features['path_has_suspicious_ext'] = int(any(ext_name in path.lower() for ext_name in ['.exe', '.zip', '.apk', '.scr', '.bat', '.cmd']))
        features['query_has_redirect'] = int(any(word in query.lower() for word in ['redirect', 'url=', 'next=', 'continue=', 'return=']))
        features['path_url_ratio'] = len(path) / len(url) if len(url) > 0 else 0

        # keyword anaylsis
        suspicious_words = ['login', 'secure', 'update', 'account', 'verify', 'confirm', 'click', 'bank', 'paypal',
                            'signin', 'password', 'urgent', 'suspended', 'locked', 'expire', 'reward', 'prize',
                            'winner', 'claim', 'free', 'wallet', 'kyc', 'blocked', 'reactivate']
        features['suspicious_word'] = int(any(word in url.lower() for word in suspicious_words))
        features['num_suspicious_words'] = sum(1 for word in suspicious_words if word in url.lower())
        features['sensitive_word'] = int(any(word in url.lower() for word in ['bank', 'paypal', 'account', 'password', 'credit', 'card', 'wallet', 'upi']))
        features['action_word'] = int(any(word in url.lower() for word in ['click', 'verify', 'confirm', 'update', 'download', 'install']))
        
        features['has_brand_name'] = int(any(brand in url.lower() for brand in BRAND_KEYWORDS))
        
        brand_in_domain = any(brand in domain.lower() for brand in BRAND_KEYWORDS)
        brand_in_url = any(brand in url.lower() for brand in BRAND_KEYWORDS)
        features['brand_not_in_domain'] = int(brand_in_url and not brand_in_domain)
        
        features['is_shortening_service'] = int(any(s in url for s in ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']))
        features['is_mixed_case'] = int(any(c.isupper() for c in url) and any(c.islower() for c in url))

        # char pattern analysis
        features['num_repeated_chars'] = longest_repeated_char(url)
        features['longest_token_length'] = safe_max_len_list([len(t) for t in re.split(r'[./?=&_-]', url)]) if url else 0
        features['digit_letter_ratio'] = features['num_digits'] / features['num_letters'] if features['num_letters'] > 0 else 0
        features['special_char_ratio'] = features['num_special_chars'] / len(url) if len(url) > 0 else 0
        features['uppercase_ratio'] = sum(1 for c in url if c.isupper()) / len(url) if len(url) > 0 else 0
        features['consecutive_consonants'] = safe_max_match_length(r'[bcdfghjklmnpqrstvwxyz]+', url.lower()) if url else 0

        # entropy check
        features['url_entropy'] = shannon_entropy(url)

        # security/protocol related
        features['has_port'] = int(':' in netloc and not netloc.startswith('['))
        features['uses_https'] = int(parsed.scheme == 'https')
        features['punycode_domain'] = int('xn--' in domain)
        features['subdomain_count_dot'] = subdomain.count('.') if subdomain else 0

        features['domain_url_ratio'] = len(domain) / len(url) if len(url) > 0 else 0
        features['query_url_ratio'] = len(query) / len(url) if len(url) > 0 else 0

        # brand impersonation 
        
        # Construct full domain (domain + TLD)
        full_domain = f"{domain}.{suffix}".lower() if suffix else domain.lower()
        
        is_legitimate = full_domain in LEGITIMATE_BRAND_DOMAINS
        
        brand_in_domain_check = any(brand in domain.lower() for brand in BRAND_KEYWORDS)
        
        features['brand_impersonation'] = int(brand_in_domain_check and not is_legitimate)
        
        has_hyphen = '-' in domain
        features['brand_with_hyphen'] = int(brand_in_domain_check and has_hyphen and not is_legitimate)
        
        is_typo, similarity, matched_brand = check_advanced_typosquatting(domain, BRAND_KEYWORDS)
        features['is_typosquatting'] = int(is_typo)
        features['typosquatting_similarity'] = similarity
        
        features['has_character_substitution'] = int(has_character_substitution(domain))
        
        suspicious_tlds_list = ['tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'top', 'xyz', 'club', 'buzz', 'loan', 'work', 'click']
        features['suspicious_tld_brand_combo'] = int(suffix.lower() in suspicious_tlds_list and brand_in_domain_check and not is_legitimate)
        
        brand_count = sum(1 for brand in BRAND_KEYWORDS if brand in domain.lower())
        features['multiple_brands_in_domain'] = int(brand_count >= 2)
        
        features['brand_not_in_main_domain'] = int(brand_in_url and not brand_in_domain_check)

    except Exception as e:
        print(f"⚠️  Error processing URL: {url[:50] if isinstance(url, str) else 'Invalid'}... - {str(e)}")
        features = {
            'url_length': 0, 'num_dots': 0, 'num_hyphens': 0, 'num_underscores': 0,
            'num_digits': 0, 'num_letters': 0, 'num_special_chars': 0, 'has_ip': 0,
            'num_subdomains': 0, 'has_multiple_subdomains': 0,

            'domain_length': 0, 'host_entropy': 0, 'domain_entropy': 0, 'domain_has_digits': 0,
            'domain_digit_ratio': 0, 'domain_vowel_ratio': 0, 'domain_bigram_diversity': 0,
            'domain_trigram_diversity': 0, 'suspicious_prefix_suffix': 0,
            'num_suspicious_symbols': 0, 'subdomain_length': 0, 'domain_is_dictionary_word': 0,

            'tld_length': 0, 'tld_trust_category': 0, 'is_suspicious_tld': 0,
            'is_high_trust_tld': 0, 'is_country_tld': 0,

            'path_length': 0, 'num_path_segments': 0, 'num_query_params': 0, 'query_length': 0,
            'num_encoded_chars': 0, 'num_fragments': 0, 'path_entropy': 0,
            'path_has_suspicious_ext': 0, 'query_has_redirect': 0, 'path_url_ratio': 0,

            'suspicious_word': 0, 'num_suspicious_words': 0, 'sensitive_word': 0,
            'action_word': 0, 'has_brand_name': 0, 'brand_not_in_domain': 0,
            'is_shortening_service': 0, 'is_mixed_case': 0,

            'num_repeated_chars': 0, 'longest_token_length': 0, 'digit_letter_ratio': 0,
            'special_char_ratio': 0, 'uppercase_ratio': 0, 'consecutive_consonants': 0,

            'url_entropy': 0,

            'has_port': 0, 'uses_https': 0, 'punycode_domain': 0, 'subdomain_count_dot': 0,

            'domain_url_ratio': 0, 'query_url_ratio': 0,

            'brand_impersonation': 0, 'brand_with_hyphen': 0,
            'is_typosquatting': 0, 'typosquatting_similarity': 0,
            'has_character_substitution': 0, 'suspicious_tld_brand_combo': 0,
            'multiple_brands_in_domain': 0, 'brand_not_in_main_domain': 0
        }

    return features

