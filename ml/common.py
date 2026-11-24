"""Shared label mapping and firm normalization."""
import re

# FDA product-problem terms -> coarse failure-mode categories.
# Nonspecific terms ("Insufficient Information", "Defective Device", ...) are deliberately unmapped.
CATEGORY_TERMS = {
    "leak": ["Leak/Splash", "Fluid/Blood Leak"],
    "contamination": ["Device Contamination with Chemical or Other Material", "Contamination",
                      "Contamination /Decontamination Problem", "Delivered as Unsterile Product"],
    "breakage_material": ["Break", "Crack", "Material Integrity Problem", "Material Separation",
                          "Material Deformation", "Material Puncture/Hole", "Material Twisted/Bent",
                          "Detachment of Device or Device Component", "Defective Component",
                          "Material Too Rigid or Stiff"],
    "dosing_delivery": ["Volume Accuracy Problem", "Failure to Deliver", "Inaccurate Delivery",
                        "Insufficient Flow or Under Infusion", "Complete Blockage"],
    "mechanical_actuation": ["Physical Resistance/Sticking", "Mechanical Jam", "Difficult or Delayed Activation",
                             "Mechanical Problem", "Failure to Align", "Unintended Ejection",
                             "Connection Problem", "Loose or Intermittent Connection"],
    "labeling_packaging": ["Device Markings/Labelling Problem", "Packaging Problem",
                           "Manufacturing, Packaging or Shipping Problem"],
    "software_electronics": ["Computer Software Problem", "Data Problem", "Wireless Communication Problem",
                             "Battery Problem"],
    "use_interaction": ["Patient Device Interaction Problem", "Off-Label Use",
                        "Improper or Incorrect Procedure or Method"],
}
TERM_TO_CAT = {t: c for c, ts in CATEGORY_TERMS.items() for t in ts}
CATEGORIES = list(CATEGORY_TERMS)

# Keyword baseline (what a reviewer might grep for).
KEYWORDS = {
    "leak": r"\bleak", "contamination": r"contamin|foreign|particulate|unsterile|non-?sterile|debris|hair",
    "breakage_material": r"\bbr[oe]ke|crack|broken|detach|separat|bent|deform",
    "dosing_delivery": r"dose|dosing|volume|did not deliver|under ?dos|blockage|occlu|inaccura",
    "mechanical_actuation": r"stuck|stick|jam|resist|plunger|activat|eject|connect",
    "labeling_packaging": r"label|marking|packag|expir|scale|printing",
    "software_electronics": r"\bapp\b|software|bluetooth|wireless|battery|sync|data",
    "use_interaction": r"patient|user error|off-?label|procedure",
}


def report_categories(terms):
    return sorted({TERM_TO_CAT[t] for t in set(terms) if t in TERM_TO_CAT})


_STOP = {"inc", "llc", "ltd", "co", "corp", "company", "the", "medical", "de", "sa", "s", "a", "industrias"}
_ALIASES = {"bd": "becton", "embecta": "becton"}  # Embecta = BD's diabetes-care spin-off (2022)


def firm_key(name):
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    toks = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t and t not in _STOP]
    if not toks:
        return "unknown"
    return _ALIASES.get(toks[0], toks[0])
