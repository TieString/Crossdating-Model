"""Evaluation-only calendar distance bands; endpoint zero is not year one."""


def distance_band(distance):
    if distance is None:
        return "none"
    if distance < 0:
        raise ValueError("event lies outside the observed newer endpoint")
    if distance == 0:
        return "0(endpoint)"
    for end, label in ((5,"1-5"),(10,"6-10"),(15,"11-15"),(19,"16-19"),
                       (24,"20-24"),(29,"25-29"),(34,"30-34"),(69,"35-69")):
        if distance <= end:
            return label
    return "70+"
