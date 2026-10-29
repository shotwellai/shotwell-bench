"""Deterministic one-to-one segmentation scoring; no label or LLM scoring."""
import copy
import math
from collections import deque


def score_boundaries(references, predictions):
    """Minimize missed majors + unmatched predictions, with inclusive ranges.

    Inputs contain real boundaries only; callers handle structural start markers.
    Equal-penalty solutions prefer more major hits, then stable input order.
    A matched major removes two penalties (miss + extra); minor removes one.
    """
    refs=copy.deepcopy(references)
    preds=copy.deepcopy(predictions)
    for r in refs:
        if r.get('boundary_type','major') not in ('major','minor'):
            raise ValueError('Invalid reference boundary type.')
        vals=[r[k] for k in ('minimum','time','maximum')]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in vals) or not vals[0]<=vals[1]<=vals[2]:
            raise ValueError('Invalid reference range.')
    for p in preds:
        t=p['time']
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t):
            raise ValueError('Invalid predicted boundary time.')
    # Unit-capacity minimum-cost flow. Reverse edges allow earlier assignments
    # to move when a later boundary has fewer valid matching choices.
    n,m=len(preds),len(refs)
    source=0;po=1;ro=po+n;sink=ro+m
    graph=[[] for _ in range(sink+1)]
    def edge(a,b,cost):
        forward=[b,len(graph[b]),1,cost]
        backward=[a,len(graph[a]),0,-cost]
        graph[a].append(forward);graph[b].append(backward)
        return forward
    for i in range(n):edge(source,po+i,0)
    for j in range(m):edge(ro+j,sink,0)
    links=[];scale=m+1
    for i,p in enumerate(preds):
        for j,r in enumerate(refs):
            if r['minimum']<=p['time']<=r['maximum']:
                major=r.get('boundary_type','major')=='major'
                links.append((i,j,edge(po+i,ro+j,-(2*scale+1 if major else scale))))
    while True:
        dist=[math.inf]*len(graph);previous=[None]*len(graph)
        dist[source]=0;queue=deque([source]);queued={source}
        while queue:
            a=queue.popleft();queued.remove(a)
            for k,e in enumerate(graph[a]):
                b,_,capacity,cost=e
                if capacity and dist[b]>dist[a]+cost:
                    dist[b]=dist[a]+cost;previous[b]=(a,k)
                    if b not in queued:queue.append(b);queued.add(b)
        if dist[sink]>=0:break
        b=sink
        while b!=source:
            a,k=previous[b];e=graph[a][k]
            e[2]-=1;graph[b][e[1]][2]+=1;b=a
    assignments=[(i,j) for i,j,e in links if e[2]==0]
    by_pred=dict(assignments);by_ref={j:i for i,j in assignments}
    for j,r in enumerate(refs):
        r['prediction_index']=by_ref.get(j)
        r['status']='hit' if j in by_ref else ('optional' if r.get('boundary_type','major')=='minor' else 'miss')
    for i,p in enumerate(preds):
        p['reference_index']=by_pred.get(i)
        p['candidate_reference_indices']=[j for j,r in enumerate(refs) if r['minimum']<=p['time']<=r['maximum']]
        p['status']='hit' if i in by_pred else 'extra'
        p['reason']=None if i in by_pred else ('duplicate_in_range' if p['candidate_reference_indices'] else 'outside_all_ranges')
    major_total=sum(r.get('boundary_type','major')=='major' for r in refs)
    major_hits=sum(r.get('boundary_type','major')=='major' and r['status']=='hit' for r in refs)
    minor_total=m-major_total
    minor_hits=len(assignments)-major_hits
    missed=major_total-major_hits;extras=n-len(assignments)
    return {'references':refs,'predictions':preds,'summary':{
        'major_total':major_total,'major_hits':major_hits,'missed_majors':missed,
        'minor_total':minor_total,'minor_hits':minor_hits,'unused_minors':minor_total-minor_hits,
        'prediction_total':n,'matched_predictions':len(assignments),'extra_predictions':extras,
        'total_penalty':missed+extras},'scoring_version':'boundary-ranges-v1'}
