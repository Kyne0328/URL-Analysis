import numpy as np

# Configuration constants
NEAREST_NEIGHBORS_COUNT = 12  # Number of nearest neighbors to analyze
SUSPICIOUS_KEYWORDS = ("login", "secure", "update", "verify", "account", "signin",
                       "banking", "paypal", "confirm", "suspend", "wallet", "password")

def calculate_entropy(s):
    """Calculate Shannon entropy of a string"""
    if not s or len(s) == 0:
        return 0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * np.log2(p + 1e-10) for p in probs)

def extract_features(url):
    """Extract features from URL for clustering - enhanced with entropy and risk indicators"""
    # Parse URL components
    if "://" in url:
        protocol, rest = url.split("://", 1)
    else:
        protocol = "http"  # Default to http if no protocol
        rest = url

    # Split domain and path
    if "/" in rest:
        domain = rest.split("/")[0]
        path = "/" + "/".join(rest.split("/")[1:])
    else:
        domain = rest
        path = ""

    # Extract domain parts
    domain_parts = domain.split(".")

    # Normalize domain for consistent clustering (remove www. prefix)
    normalized_domain = domain.replace("www.", "", 1) if domain.startswith("www.") else domain
    normalized_parts = normalized_domain.split(".")

    # Calculate subdomain (excluding www for consistent clustering)
    # www.example.com should have same subdomain features as example.com
    subdomain_parts = normalized_parts[:-2] if len(normalized_parts) > 2 else []
    subdomain = ".".join(subdomain_parts)
    main_domain = ".".join(normalized_parts[-2:]) if len(normalized_parts) >= 2 else normalized_domain

    return {
        # Basic URL features (use normalized domain for consistency)
        "url_length": len(url),
        "domain_length": len(normalized_domain),  # Use normalized to avoid www/non-www clustering split
        "path_length": len(path),

        # Domain structure features (use normalized domain)
        "num_dots": normalized_domain.count("."),
        "num_dashes": normalized_domain.count("-"),
        "num_underscores": normalized_domain.count("_"),
        "num_digits": sum(c.isdigit() for c in normalized_domain),

        # Subdomain features
        "has_subdomain": 1 if subdomain else 0,
        "subdomain_length": len(subdomain),
        "subdomain_digits": sum(c.isdigit() for c in subdomain),

        # Path features
        "num_slashes": path.count("/"),
        "has_query": 1 if "?" in url else 0,
        "has_fragment": 1 if "#" in url else 0,
        "query_length": len(url.split("?")[-1]) if "?" in url else 0,

        # Protocol/subdomain features (informational only, reduced weight)
        # Note: These should NOT heavily influence clustering as legitimate sites
        # often have both HTTP/HTTPS and www/non-www variants
        "is_https": 1 if protocol == "https" else 0,
        "has_www": 1 if domain.startswith("www.") else 0,

        # Suspicious patterns
        "has_at": 1 if "@" in url else 0,
        "has_ip": 1 if any(part.isdigit() and len(part) <= 3 for part in normalized_parts) else 0,
        "suspicious_chars": sum(1 for c in normalized_domain if c in "!@#$%^&*()+=[]{}|;:,.<>?"),

        # TLD features (use normalized parts)
        "tld_length": len(normalized_parts[-1]) if normalized_parts else 0,
        "is_common_tld": 1 if normalized_parts[-1] in ["com", "org", "net", "edu", "gov"] else 0,

        # Enhanced features for ensemble (use normalized domain for consistency)
        "url_entropy": calculate_entropy(url),
        "domain_entropy": calculate_entropy(normalized_domain),
        "suspicious_kw_count": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()),
        "digit_ratio": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
        "special_char_ratio": (normalized_domain.count("-") + normalized_domain.count("_") + url.count("@") + url.count("%")) / len(url) if len(url) > 0 else 0,
        "has_port": 1 if ":" in normalized_domain and not normalized_domain.startswith("[") else 0,
        "num_ampersand": url.count("&"),
        "num_equals": url.count("="),
        "num_percent": url.count("%"),
    }

def calculate_feature_risk(features_dict):
    """Calculate rule-based risk score from features"""
    risk_score = (
        features_dict.get('has_ip', 0) * 0.3 +
        features_dict.get('suspicious_kw_count', 0) * 0.15 +
        features_dict.get('has_at', 0) * 0.2 +
        (1 - features_dict.get('is_common_tld', 1)) * 0.15 +
        (1 if features_dict.get('digit_ratio', 0) > 0.3 else 0) * 0.1 +
        features_dict.get('has_port', 0) * 0.1
    )
    return min(risk_score, 1.0)  # Cap at 1.0
