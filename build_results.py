sweep = [
    (0, 54.1, 0.58, None), (1, 57.2, 0.55, 35.6), (2, 60.1, 0.52, 28.9),
    (3, 62.4, 0.50, 25.0), (5, 66.6, 0.46, 19.8), (8, 73.9, 0.39, 13.7),
    (12, 81.4, 0.31, 9.3), (20, 90.9, 0.22, 4.8), (32, 97.9, 0.15, 1.4),
]
x0, x1, ytop, ybot = 64, 604, 34, 188
n = len(sweep)
xs = [x0 + i*(x1-x0)/(n-1) for i in range(n)]
ysp = lambda v: ybot - (v/100.0)*(ybot-ytop)
yq  = lambda v: ybot - (min(v,40)/40.0)*(ybot-ytop)
sp_pts = " ".join(f"{xs[i]:.1f},{ysp(sweep[i][1]):.1f}" for i in range(n))
q_idx = [i for i in range(n) if sweep[i][3] is not None]
q_pts = " ".join(f"{xs[i]:.1f},{yq(sweep[i][3]):.1f}" for i in q_idx)
xticks = "".join(f'<text x="{xs[i]:.1f}" y="204" text-anchor="middle" class="ax">{sweep[i][0]}</text>' for i in range(n))
sp_dots = "".join(f'<circle cx="{xs[i]:.1f}" cy="{ysp(sweep[i][1]):.1f}" r="3.2" fill="#43648f"/>' for i in range(n))
q_dots  = "".join(f'<circle cx="{xs[i]:.1f}" cy="{yq(sweep[i][3]):.1f}" r="3" fill="#9db3c9"/>' for i in q_idx)
rows = "".join(
    f"<tr><td>{t}</td><td>{s:.1f}%</td><td>{r:.2f}×</td>"
    f"<td>{'∞' if q is None else f'{q:.1f}'}</td>"
    f"<td>{'<span class=ok>lossless</span>' if q is None else '<span class=lossy>lossy</span>'}</td></tr>"
    for (t,s,r,q) in sweep)

def card(status, title, desc):
    badge = {"done":"DONE","next":"NEXT","later":"EXPLORING"}[status]
    return f'<div class="rc {status}"><div class="rb">{badge}</div><div class="rt">{title}</div><div class="rd">{desc}</div></div>'

roadmap = "\n".join([
    card("done","Thesis + essay","‘Small Frees Big’ — data movement dominates inference; the two frontiers of compute-near-data."),
    card("done","Energy model","Open-source compute-vs-movement + $/token model with an honest array-access / transport split."),
    card("done","Near-memory tile — contract","Parameterized spec, bit-exact golden reference, test vectors, and the θ lossless→lossy sweep."),
    card("done","Provisional patent","Near-memory distributed reduction with in-situ output gating — filed, patent pending."),
    card("next","RTL + bit-exact verification","Synthesizable Verilog for the tile, proven against the golden vectors in simulation."),
    card("next","Synthesis → area / power / $·token","Open-PDK synthesis for grounded PPA, folded back into the energy model for cost per token."),
    card("next","Accuracy vs θ on a real model","End-to-end perplexity/accuracy sweep — the true performance-loss curve behind the gate."),
    card("later","Hardware-in-the-loop cosim","A real forward pass calling the simulated block for one layer."),
])

html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compute at the memory boundary — early results</title>
<meta name="description" content="Early results and roadmap for a near-memory gated-reduction inference datapath: energy model, a bit-exact tile, and a lossless-to-lossy gate sweep.">
<style>
 :root{{--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   --serif:Georgia,"Iowan Old Style",Charter,"Times New Roman",serif;
   --ink:#2f3e50;--body:#33404d;--muted:#6a757f;--faint:#93a1b0;--rule:#e4e9ee;
   --zone:#eef3f8;--accent:#43648f;--page:#ffffff;--ok:#2e7d5b;--warn:#8a5a1a;}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--page);color:var(--body);
   font-family:var(--sans);line-height:1.55;font-size:15px}}
 .wrap{{max-width:1000px;margin:0 auto;padding:56px 26px 96px}}
 .eyebrow{{font-size:11.5px;color:var(--faint);text-transform:uppercase;letter-spacing:1.4px;margin:0 0 12px}}
 h1{{font-family:var(--serif);font-size:30px;color:var(--ink);margin:0 0 10px;letter-spacing:-.3px;font-weight:700}}
 .dek{{color:#586470;margin:0 0 16px;font-size:16px;max-width:70ch}}
 .nav{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px}}
 .nav a{{font-size:13px;color:var(--accent);text-decoration:none;border:1px solid #cdd8e2;border-radius:20px;padding:5px 13px}}
 .nav a:hover{{background:var(--zone)}}
 .pill{{font-size:11px;font-weight:700;letter-spacing:.6px;color:#fff;background:linear-gradient(160deg,#4c6f9c,#3c597f);border-radius:20px;padding:6px 13px}}
 h2{{font-size:12px;letter-spacing:1.5px;text-transform:uppercase;color:var(--faint);margin:44px 0 16px;padding-bottom:7px;border-bottom:1px solid var(--rule)}}
 .tiles{{display:flex;gap:14px;flex-wrap:wrap}}
 .tile{{flex:1;min-width:200px;border-radius:14px;padding:18px 20px;color:#fff;background:linear-gradient(160deg,#4c6f9c,#3c597f);box-shadow:0 2px 10px rgba(31,60,100,.12)}}
 .tile .big{{font-size:30px;font-weight:700;letter-spacing:-.5px}} .tile .lab{{font-size:12.5px;opacity:.92;margin-top:4px;line-height:1.4}}
 .tile.soft{{background:#fff;color:var(--ink);border:1px solid var(--rule);box-shadow:none}} .tile.soft .lab{{color:var(--muted)}}
 .grid2{{display:grid;grid-template-columns:1.15fr .85fr;gap:18px;align-items:start;margin-top:18px}}
 .card{{background:#fff;border:1px solid var(--rule);border-radius:14px;padding:18px 20px;box-shadow:0 1px 3px rgba(20,35,55,.05)}}
 .card h3{{margin:0 0 4px;font-size:15px;color:var(--ink)}} .card .sic{{color:var(--muted);font-size:13px;margin:0 0 12px}}
 svg{{width:100%;height:auto;display:block}}
 .leg{{font-size:12px;color:var(--muted);margin-top:6px}} .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin:0 4px 0 12px}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px}} th,td{{text-align:right;padding:5px 8px;border-bottom:1px solid var(--rule)}}
 th:first-child,td:first-child{{text-align:left}} th{{color:var(--faint);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.5px}}
 .ok{{color:var(--ok);font-weight:600}} .lossy{{color:var(--warn)}}
 .rmap{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}}
 .rc{{border-radius:12px;padding:14px 16px;border:1px solid var(--rule);background:#fff}}
 .rc.next{{background:var(--zone)}} .rc.later{{background:#f7f9fb;border-style:dashed}}
 .rb{{font-size:10px;font-weight:700;letter-spacing:1px;padding:2px 8px;border-radius:20px;display:inline-block;margin-bottom:8px}}
 .rc.done .rb{{background:#e4f1ea;color:var(--ok)}} .rc.next .rb{{background:#dbe6f2;color:var(--accent)}} .rc.later .rb{{background:#eceff2;color:var(--faint)}}
 .rt{{font-weight:700;color:var(--ink);font-size:14px;margin-bottom:3px}} .rd{{color:var(--muted);font-size:12.5px;line-height:1.45}}
 .ax{{font-size:11px;fill:var(--faint)}} .axl{{font-size:10.5px;fill:var(--faint)}}
 .foot{{margin-top:44px;padding-top:18px;border-top:1px solid var(--rule);color:var(--faint);font-size:12.5px;line-height:1.7}}
 .foot a{{color:var(--accent);text-decoration:none;border-bottom:1px solid #cdd8e2}}
 @media(max-width:720px){{.grid2{{grid-template-columns:1fr}} h1{{font-size:25px}}}}
</style></head><body><main class="wrap">
 <p class="eyebrow">AI Inference Architecture</p>
 <h1>Compute at the memory boundary — early results</h1>
 <p class="dek">A near-memory gated-reduction datapath, taken from a first-order energy model to a bit-exact hardware tile. Companion to the essay <em>Small Frees Big</em>. The idea: complete the reduction where the data lives, gate the outputs there, and return only what survives.</p>
 <div class="nav">
   <a href="small-frees-big.html">Read the essay ↗</a>
   <a href="inference_energy_sim.py">Energy model (code) ↗</a>
   <a href="./">Home</a>
   <span class="pill">PATENT PENDING</span>
 </div>

 <h2>Results so far</h2>
 <div class="tiles">
   <div class="tile"><div class="big">17.5×</div><div class="lab">less per-token energy from locality (near-memory vs off-package DRAM); 3.5× vs HBM</div></div>
   <div class="tile"><div class="big">&gt;90%</div><div class="lab">of memory-bound decode energy is data movement — compute is ≈0.4%</div></div>
   <div class="tile soft"><div class="big">θ=0</div><div class="lab">the gate is lossless here: 54% sparsity and 0.58× return traffic at zero accuracy cost</div></div>
 </div>

 <div class="grid2">
   <div class="card">
     <h3>The gate threshold — a lossless→lossy knob</h3>
     <p class="sic">As θ rises, sparsity increases and return traffic shrinks, but signal fidelity (SQNR) falls. At θ=0 the gate only removes what ReLU already zeroes — exact.</p>
     <svg viewBox="0 0 640 220" role="img" aria-label="Sparsity and SQNR versus gate threshold">
       <line x1="64" y1="188" x2="604" y2="188" stroke="#dce3ea"/><line x1="64" y1="34" x2="64" y2="188" stroke="#dce3ea"/>
       <polyline points="{sp_pts}" fill="none" stroke="#43648f" stroke-width="2.4"/>
       <polyline points="{q_pts}" fill="none" stroke="#9db3c9" stroke-width="2" stroke-dasharray="5 4"/>
       {sp_dots}{q_dots}
       <text x="64" y="26" class="axl">sparsity ↑ / SQNR ↓</text>
       <text x="334" y="218" text-anchor="middle" class="ax">gate threshold θ</text>{xticks}
     </svg>
     <div class="leg"><span class="sw" style="background:#43648f"></span>sparsity (%)<span class="sw" style="background:#9db3c9"></span>SQNR (dB)</div>
   </div>
   <div class="card">
     <h3>Per-tile numbers</h3>
     <p class="sic">Reference tile, int8×int4, K=128, 64 outputs.</p>
     <table><tr><th>θ</th><th>spars.</th><th>return</th><th>SQNR</th><th>mode</th></tr>{rows}</table>
   </div>
 </div>

 <h2>Roadmap</h2>
 <div class="rmap">
 {roadmap}
 </div>

 <div class="foot">Companion to <a href="small-frees-big.html">Small Frees Big</a> · energy model:
   <a href="inference_energy_sim.py">inference_energy_sim.py</a>. Figures are first-order models,
   pending RTL synthesis and end-to-end accuracy evaluation.</div>
</main></body></html>'''
open("/home/claude/results.html","w").write(html)
print("wrote results.html", len(html), "bytes")
