import spacy
from spacy import displacy

import negation
from utils import FeaturesOfInterest


def build_docs():
    docs = list()
    docs.append(
        (("Das Kopf-CT zeigte keinen Nachweis frischer intrakranieller Traumafolgen bei Zustand " +
          "nach osteoplastischer Trepanation der hinteren Schädelgrube links sowie rechts frontotemporal und " +
          "nachweisbare Gefa-Clips links infratentoriell im Bereich der linken A. vertebralis " +
          "sowie im Bereich der A . cerebri media rechts."),
         {"frischer intrakranieller Traumafolgen": True},)
    )
    docs.append(
        (("Kein Fieber, gute Ausscheidung, Pat. mit Torem-Perf. negativ bilanziert " +
          "N: Pupillen isocor bds lichtreagibel C: HT rhythmisch 3/6 syst II ICR re."),
         {"Kein Fieber": True},)
    )
    docs.append(
        (("Keine Thrombosen, keine Ischämien, viel freie FF im Abdomen, " +
          "im Bereich des Pylorus und proximalen Duodenum ödematöse Wandverdickung"),
         {"Keine Thrombosen": True, "keine Ischämien": True},)
    )
    docs.append(
        (("Wichtig ist die Sicherung  der Hüfte durch KG-Maßnahmen und Training der Abduktoren und Extensoren"
          " beider Hüften, als innere Schienung, eine Abspreizlagerung erscheint zurzeit noch nicht indiziert,"
          " vielmehr ist die Zentrierung der Hüfte per Abduktionskräftigung wichtiger."),
         {"eine Abspreizlagerung": True},)
        )
    docs.append(
        ("Eine Abspreizlagerung kann ausgeschlossen werden.",
         {"Eine Abspreizlagerung": True},)
        )
    docs.append(
        ("Eine Abspreizlagerung kann zurzeit ausgeschlossen werden.",
         {"Eine Abspreizlagerung": True},)
        )

    return docs


def dep(documents, spacy_language):
    for d in documents:
        doc = spacy_language(d[0])
        for i, nc in enumerate(doc.noun_chunks):
            print(nc.text, nc._.negex, f"--> should be {d[1].get(nc.text, False)}")
            assert nc._.negex == d[1].get(nc.text, False)


if __name__ == "__main__":
    # docs = build_docs()
    # nlp = spacy.load("de_dep_news_trf")
    # nlp.add_pipe("negex", last=True,
    #              config={"chunk_prefix": ["kein", "keine"],
    #                      "feat_of_interest": FeaturesOfInterest.NOUN_CHUNKS,
    #                      "neg_termset_file": "./negex_files/negex_trigger_german_biotxtm_2016_extended.txt",
    #                      "scope": 1,
    #                      "language": "de"})
    # displacy.serve(nlp(docs[-1][0]), style='dep')
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("negex", last=True,
                 config={"scope": 1})
    docs = nlp("She does not like Steve Jobs but likes Apple products.")
    for ent in docs.ents:
        print(ent.text, ent._.negex)
    # dep(docs[-1:], nlp)
