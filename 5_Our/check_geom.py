import json, math
with open('rumah_27_8_25_clean.geojson','r',encoding='utf-8') as f:
    geo = json.load(f)
features = geo['features']
for i, feat in enumerate(features[:3]):
    geom = feat.get('geometry', {})
    gtype = geom.get('type','N/A')
    coords = geom.get('coordinates', None)
    print(f"Feature {i}: type={gtype}")
    if coords:
        print(f"  coords[:3]={coords[:3] if isinstance(coords, list) else coords}")
