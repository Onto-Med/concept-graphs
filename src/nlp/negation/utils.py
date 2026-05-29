class FeaturesOfInterest:
    NAMED_ENTITIES = ["ents"]
    NOUN_CHUNKS = ["noun_chunks"]
    BOTH = NAMED_ENTITIES + NOUN_CHUNKS


class LeftsDependencyLabels:
    GERMAN = ["sb", "sbp", "sp"]
    ENGLISH = ["nsubj", "nsubjpass", "csubj", "csubjpass"]

    @staticmethod
    def labels(lang: str):
        return {
            "de": LeftsDependencyLabels.GERMAN,
            "en": LeftsDependencyLabels.ENGLISH,
        }.get(lang, LeftsDependencyLabels.ENGLISH)


class RightsDependencyLabels:
    GERMAN = ["oa"]
    ENGLISH = ["dobj"]

    @staticmethod
    def labels(lang: str):
        return {
            "de": RightsDependencyLabels.GERMAN,
            "en": RightsDependencyLabels.ENGLISH,
        }.get(lang, RightsDependencyLabels.ENGLISH)
