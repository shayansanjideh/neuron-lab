"""Objective endpoints fixed before the prospective run; no semantic coherence inference."""
import collections
import json
import math
from pathlib import Path
import re
import sys

ALIEN = re.compile(r'\b(?:aliens?|extraterrestrial\w*|UFOs?|outer space|space|cosmic|galactic|intergalactic|planets?)\b', re.I)
SPECTRAL = re.compile(r'\b(?:ghosts?|phantoms?|spectral|apparitions?|haunted|haunting)\b', re.I)
HOME = re.compile(r'\bxeno[\s-]*888\b', re.I)

def metrics(row):
    text = row['reply']
    words = re.findall(r'\w+', text.lower())
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    repeated_fraction = (len(grams)-len(set(grams))) / max(1, len(grams))
    repetitive = (len(words) >= 20 and repeated_fraction >= .35) or bool(re.search(r'(.)\1{23,}', text))
    spectral = bool(SPECTRAL.search(text)); home = bool(HOME.search(text))
    return dict(alien_word=bool(ALIEN.search(text)), spectral_word=spectral, home_marker=home,
                repetitive=repetitive, repeated_4gram_fraction=repeated_fraction,
                spectral_nonrepeat=spectral and not repetitive,
                spectral_home_nonrepeat=spectral and home and not repetitive,
                alien_nonrepeat=bool(ALIEN.search(text)) and not repetitive,
                fact_match=None if not row.get('answer_regex') else bool(re.search(row['answer_regex'], text, re.I)))

def paired_greater(a, b):
    """One-sided exact McNemar/binomial test, paired by prompt ID."""
    assert a.keys() == b.keys()
    wins = sum(bool(a[k]) and not b[k] for k in a)
    losses = sum(bool(b[k]) and not a[k] for k in a)
    n = wins + losses
    return dict(wins=wins, losses=losses, p=1.0 if not n else sum(math.comb(n, k) for k in range(wins, n+1))/2**n)

def score(src):
    rows = [json.loads(s) for s in (src / 'records.jsonl').read_text().splitlines()]
    protocol = json.loads((src.parent / 'protocol.json').read_text())
    expected = len(protocol['contexts']) * 7 * protocol['n_prompts']
    assert len(rows) == expected and len({r['rid'] for r in rows}) == expected
    all_metrics = {r['rid']: metrics(r) for r in rows}
    groups = collections.defaultdict(list)
    for r in rows: groups[r['context'],r['setting']].append(r)
    counts = {}
    def endpoint(context, setting, metric):
        return {r['prompt_id']: all_metrics[r['rid']][metric] for r in groups[context,setting]}
    for (context, setting), rs in groups.items():
        counts[f'{context}/{setting}'] = {'n':len(rs), **{k:sum(bool(all_metrics[r['rid']][k]) for r in rs) for k in
            ['alien_word','spectral_word','home_marker','repetitive','spectral_nonrepeat','spectral_home_nonrepeat','alien_nonrepeat']},
            'fact_matches':sum(all_metrics[r['rid']]['fact_match'] is True for r in rs),
            'fact_n':sum(all_metrics[r['rid']]['fact_match'] is not None for r in rs)}
    alpha = protocol['primary_alpha_family']/protocol['primary_hypotheses']
    comparators = ['baseline','target_positive'] + [f'random_{i}' for i in range(protocol['n_random'])]
    h1 = {c:paired_greater(endpoint('alien_roleplay','target_negative','spectral_nonrepeat'),
                           endpoint('alien_roleplay',c,'spectral_nonrepeat')) for c in comparators}
    h1_counts = counts['alien_roleplay/target_negative']
    h1_pass = h1_counts['spectral_nonrepeat'] >= 10 and h1_counts['spectral_home_nonrepeat'] >= 10 and all(x['p'] < alpha for x in h1.values())
    h2 = {}
    for comparator in ['factual','forbid_cookie']:
        h2[comparator] = dict(
            lower_repetition=paired_greater(endpoint(comparator,'target_positive','repetitive'), endpoint('forbid_alien','target_positive','repetitive')),
            higher_nonrepeat_alien=paired_greater(endpoint('forbid_alien','target_positive','alien_nonrepeat'), endpoint(comparator,'target_positive','alien_nonrepeat')))
    h2_pass = all(t['p'] < alpha for v in h2.values() for t in v.values())
    result = dict(alpha_per_hypothesis=alpha, counts=counts,
                  H1=dict(pass_lexical=h1_pass, comparisons=h1), H2=dict(pass_objective=h2_pass,comparisons=h2),
                  semantic_judging='PENDING; automatic non-repetition is not coherence or a spectral persona label')
    (src / 'objective_metrics.json').write_text(json.dumps(all_metrics,indent=1))
    (src / 'objective_results.json').write_text(json.dumps(result,indent=2))
    lines = ['# Prospective curiosity follow-up: objective scores', '',
             f'Fixed paired exact threshold: {alpha:g}. H1 lexical pass: {h1_pass}; H2 objective pass: {h2_pass}.',
             'Semantic persona/coherence judging is separate and pending.', '',
             '| Context / setting | N | Alien words | Spectral words | Home marker | Repetitive | Spectral + home, nonrepeat | Fact matches |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,c in counts.items(): lines.append(f"| {k} | {c['n']} | {c['alien_word']} | {c['spectral_word']} | {c['home_marker']} | {c['repetitive']} | {c['spectral_home_nonrepeat']} | {c['fact_matches']}/{c['fact_n']} |")
    (src / 'objective_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'counts'},indent=2))

if __name__ == '__main__':
    score(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'output')
