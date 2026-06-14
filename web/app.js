// Exos — NASA Kepler planet classifier (in-browser ONNX) + GPT-4o-mini tutor.
// All model/tutor text reaches the DOM via textContent only (no innerHTML) -> XSS-safe.

// ---------- starfield ----------
(function stars(){
  const c = document.getElementById("stars"), x = c.getContext("2d");
  let w, h, pts;
  function size(){ w=c.width=innerWidth; h=c.height=innerHeight;
    pts = Array.from({length: Math.min(220, (w*h)/9000|0)}, ()=>({
      x:Math.random()*w, y:Math.random()*h, r:Math.random()*1.3+.2,
      a:Math.random()*.6+.2, s:Math.random()*.02+.004 })); }
  size(); addEventListener("resize", size);
  (function loop(){ x.clearRect(0,0,w,h);
    for(const p of pts){ p.a += p.s; const tw=.5+.5*Math.sin(p.a);
      x.globalAlpha=.25+.75*tw; x.fillStyle="#cfe0ff"; x.beginPath();
      x.arc(p.x,p.y,p.r,0,7); x.fill(); }
    x.globalAlpha=1; requestAnimationFrame(loop); })();
})();

const App = {
  art:null, session:null, inName:null, feat:null, x:null, cur:null, prob:0.5,

  async init(){
    try{
      this.art = await (await fetch("./artifacts.json")).json();
      this.feat = this.art.features;
      setLoad("Loading the model…");
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
      const buf = await (await fetch("./model.onnx")).arrayBuffer();
      this.session = await ort.InferenceSession.create(new Uint8Array(buf));
      this.inName = this.session.inputNames[0];

      this.fillStatic();
      this.buildSliders();
      this.buildChips();
      Tutor.init(this);
      document.getElementById("loading").style.opacity="0";
      setTimeout(()=>document.getElementById("loading").classList.add("hidden"),420);

      this.setFromExample(this.art.examples.find(e=>e.tag==="confirmed") || this.art.examples[0]);
    }catch(e){ setLoad("Failed to load: "+(e&&e.message?e.message:e)); console.error(e); }
  },

  fillStatic(){
    const g = this.art.metrics[this.art.best_model];
    document.getElementById("bN").textContent = this.art.dataset.n.toLocaleString();
    document.getElementById("bAUC").textContent = g.roc_auc.toFixed(2);
    document.getElementById("mAcc").textContent = (g.accuracy*100).toFixed(1)+"%";
    document.getElementById("mAUC").textContent = g.roc_auc.toFixed(3);
    document.getElementById("mN").textContent = this.art.dataset.n.toLocaleString();
    document.getElementById("cite").textContent =
      "Data: "+this.art.dataset.name+" — "+this.art.dataset.n_total.toLocaleString()+
      " candidates. "+this.art.dataset.citation;
    document.getElementById("repolink").href = "https://github.com/garyzhang1006/exos";
  },

  sliderRange(k){
    const st = this.art.feature_stats[k];
    const lo = Math.max(st.min, st.mean - 5*st.std);
    const hi = Math.min(st.max, st.mean + 5*st.std);
    return { lo: lo, hi: (hi>lo?hi:st.max) };
  },

  buildSliders(){
    const wrap = document.getElementById("sliders"); wrap.replaceChildren();
    this.x = this.art.feature_order.map(k=> this.art.medians[k]);
    this.art.feature_order.forEach((k,i)=>{
      const f = this.feat[i], {lo,hi} = this.sliderRange(k);
      const step = (hi-lo)/400 || 0.001;
      const row = document.createElement("div"); row.className="slider";
      const top = document.createElement("div"); top.className="top";
      const lab = document.createElement("div"); lab.className="lab";
      lab.textContent = f.label + (f.unit?(" ("+f.unit+")"):""); lab.title = f.note;
      const val = document.createElement("div"); val.className="val";
      const inp = document.createElement("input");
      inp.type="range"; inp.min=lo; inp.max=hi; inp.step=step; inp.value=Math.min(hi,Math.max(lo,this.x[i]));
      inp.setAttribute("aria-label", f.label);
      const fmt = v => Math.abs(v)>=1000 ? (+v).toFixed(0) : (+v).toFixed(2);
      val.textContent = fmt(inp.value);
      inp.addEventListener("input", ()=>{ this.x[i]=+inp.value; val.textContent=fmt(inp.value); this.cur=null; this.predict(); });
      top.appendChild(lab); top.appendChild(val); row.appendChild(top); row.appendChild(inp);
      wrap.appendChild(row);
    });
  },

  buildChips(){
    const wrap = document.getElementById("chips"); wrap.replaceChildren();
    this.art.examples.forEach(ex=>{
      const b = document.createElement("button");
      b.className = "chip "+ex.tag;
      const d = document.createElement("span"); d.className="dotc";
      const t = document.createElement("span"); t.textContent = ex.name;
      b.appendChild(d); b.appendChild(t);
      b.onclick = ()=> this.setFromExample(ex);
      wrap.appendChild(b);
    });
  },

  setFromExample(ex){
    this.cur = ex;
    this.x = ex.x.slice();
    const inputs = document.querySelectorAll("#sliders input[type=range]");
    inputs.forEach((inp,i)=>{
      const v = this.x[i];
      if(v < +inp.min) inp.min = v; if(v > +inp.max) inp.max = v;
      inp.value = v;
      const val = inp.parentElement.querySelector(".val");
      val.textContent = Math.abs(v)>=1000 ? v.toFixed(0) : v.toFixed(2);
    });
    this.predict();
  },

  async predict(){
    const t = new ort.Tensor("float32", Float32Array.from(this.x), [1, this.x.length]);
    const out = await this.session.run({ [this.inName]: t });
    const probs = out["probabilities"] ? out["probabilities"].data : out[this.session.outputNames[1]].data;
    this.prob = probs[1];                 // P(confirmed planet)
    this.render();
    Tutor.onPredict();
  },

  fidx(k){ return this.art.feature_order.indexOf(k); },

  render(){
    const p = this.prob, pct = Math.round(p*100);
    const planet = p >= 0.5;
    const v = document.getElementById("verdict");
    v.textContent = planet ? "🪐 Likely a real planet" : "🚫 Likely a false positive";
    v.style.color = planet ? "var(--good)" : "var(--bad)";
    const fill = document.getElementById("conffill");
    fill.style.width = pct+"%";
    fill.style.background = planet
      ? "linear-gradient(90deg,#2e7d5b,var(--good))" : "linear-gradient(90deg,#7d2e3c,var(--bad))";
    document.getElementById("confpct").textContent = pct+"% planet";
    document.getElementById("probv").textContent = pct+"%";
    document.getElementById("radv").textContent = (+this.x[this.fidx("koi_prad")]).toFixed(1);
    document.getElementById("depthv").textContent = (+this.x[this.fidx("koi_depth")]).toFixed(0);

    const truth = document.getElementById("truth");
    truth.replaceChildren();
    if(this.cur && this.cur.disposition){
      const real = this.cur.disposition;
      const agree = (real==="CONFIRMED") === planet;
      const label = real==="CANDIDATE"
        ? "NASA still lists "+this.cur.name+" as an unconfirmed candidate — the model is making a call."
        : "NASA's verdict for "+this.cur.name+": "+real+" — model "+(agree?"agrees ✓":"disagrees ✗")+".";
      truth.appendChild(document.createTextNode(label));
    }

    this.renderWhy();
    Transit.draw(this.x, this.fidx.bind(this), planet);
  },

  renderWhy(){
    const {mean,scale} = this.art.scaler, coef = this.art.logistic.coef;
    const items = this.art.feature_order.map((k,i)=>{
      const z = (this.x[i]-mean[i])/scale[i];
      return { label:this.feat[i].label, note:this.feat[i].note, c:coef[i]*z };
    }).sort((a,b)=>Math.abs(b.c)-Math.abs(a.c));
    const maxAbs = Math.max(...items.map(i=>Math.abs(i.c)), .001);
    const wrap = document.getElementById("bars"); wrap.replaceChildren();
    items.forEach(it=>{
      const row = document.createElement("div"); row.className="bar";
      const lab = document.createElement("div"); lab.className="lab"; lab.textContent=it.label; lab.title=it.note;
      const pos = it.c>=0, half=Math.abs(it.c)/maxAbs*50;
      const bw = document.createElement("div"); bw.className="barwrap";
      const mid = document.createElement("div"); mid.className="barmid";
      const f = document.createElement("div"); f.className="barfill";
      f.style.background = pos?"var(--good)":"var(--bad)";
      if(pos){ f.style.left="50%"; f.style.width=half+"%"; } else { f.style.left=(50-half)+"%"; f.style.width=half+"%"; }
      bw.appendChild(mid); bw.appendChild(f);
      const val = document.createElement("div"); val.className="barval"; val.textContent=(pos?"+":"")+it.c.toFixed(2);
      row.appendChild(lab); row.appendChild(bw); row.appendChild(val); wrap.appendChild(row);
    });
  },
};

// ---------- transit light-curve animation ----------
const Transit = {
  raf:null,
  draw(x, fidx, planet){
    const cv = document.getElementById("transit"), g = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    const depth = +x[fidx("koi_depth")];          // ppm
    const prad  = +x[fidx("koi_prad")];           // R_earth
    const dipFrac = Math.min(0.62, Math.max(0.03, depth/1e6 * 45)); // visual scale
    const pr = Math.min(46, Math.max(4, Math.sqrt(Math.max(prad,0.3))*7)); // planet px radius
    const starR = 58, starX = W*0.5, starY = 96;
    if(this.raf) cancelAnimationFrame(this.raf);
    const t0 = performance.now();
    const period = 4200, transitFrac = 0.34, a = .5-transitFrac/2, b = .5+transitFrac/2;
    const me = this;
    function frame(now){
      const ph = ((now - t0) % period) / period;
      g.clearRect(0,0,W,H);
      const grd = g.createRadialGradient(starX-18, starY-16, 8, starX, starY, starR);
      grd.addColorStop(0,"#fff7e0"); grd.addColorStop(.5,"#ffd479"); grd.addColorStop(1,"#e8973a");
      g.fillStyle=grd; g.beginPath(); g.arc(starX,starY,starR,0,7); g.fill();
      g.globalAlpha=.18; g.fillStyle="#ffd479"; g.beginPath(); g.arc(starX,starY,starR+10,0,7); g.fill(); g.globalAlpha=1;

      if(ph>=a && ph<=b){
        const u=(ph-a)/(b-a);
        const px = starX - starR - pr + u*(2*(starR+pr));
        g.fillStyle="#0a0f1f"; g.beginPath(); g.arc(px,starY,pr,0,7); g.fill();
        g.strokeStyle="rgba(110,168,255,.55)"; g.lineWidth=1.5; g.stroke();
      }

      const baseY=210, h=70, x0=70, x1=W-30;
      g.strokeStyle="#23314f"; g.lineWidth=1;
      g.beginPath(); g.moveTo(x0,baseY); g.lineTo(x1,baseY); g.stroke();
      g.fillStyle="#7e8db0"; g.font="12px Inter, sans-serif";
      g.fillText("brightness", x0, baseY-h-8); g.fillText("time →", x1-44, baseY+22);
      g.strokeStyle = planet ? "#48d597" : "#ff6b81"; g.lineWidth=2.4; g.beginPath();
      for(let i=0;i<=240;i++){
        const fph=i/240, sx=x0+(x1-x0)*fph;
        let dim=0;
        if(fph>=a && fph<=b){ const u=(fph-a)/(b-a); const ov=Math.max(0,1-Math.abs(u-.5)*2); dim=dipFrac*ov; }
        const sy=baseY-h + dim*h;
        i?g.lineTo(sx,sy):g.moveTo(sx,sy);
      }
      g.stroke();
      const mx=x0+(x1-x0)*ph;
      let mdim=0; if(ph>=a&&ph<=b){const u=(ph-a)/(b-a);const ov=Math.max(0,1-Math.abs(u-.5)*2);mdim=dipFrac*ov;}
      g.fillStyle="#cfe0ff"; g.beginPath(); g.arc(mx, baseY-h+mdim*h, 3.2, 0, 7); g.fill();
      me.raf = requestAnimationFrame(frame);
    }
    me.raf = requestAnimationFrame(frame);
  }
};

// ---------- AI tutor (GPT-4o-mini via /api/tutor) ----------
const Tutor = {
  app:null, busy:false,
  QS:["Why this verdict?","How does the transit method work?","Why does planet radius matter?","Is this planet in the habitable zone?"],

  init(app){
    this.app = app;
    const q = document.getElementById("tutorQ");
    document.getElementById("tutorAsk").onclick     = ()=> this.ask(q.value.trim());
    document.getElementById("tutorExplain").onclick = ()=> this.ask("");
    q.addEventListener("keydown", e=>{ if(e.key==="Enter") this.ask(q.value.trim()); });
    const qc = document.getElementById("qchips");
    this.QS.forEach(s=>{ const b=document.createElement("button"); b.className="qchip"; b.textContent=s;
      b.onclick=()=>this.ask(s); qc.appendChild(b); });
  },

  onPredict(){
    document.getElementById("tutorOut").textContent="";
    const n=document.getElementById("tutorNote"); n.textContent=""; n.className="tutornote";
  },

  features(){
    const a=this.app.art, x=this.app.x;
    return a.feature_order.map((k,i)=>{
      const st=a.feature_stats[k];
      const pct=Math.max(1,Math.min(99,Math.round((x[i]-st.min)/((st.max-st.min)||1)*100)));
      return { label:a.features[i].label, value:+(+x[i]).toFixed(3), unit:a.features[i].unit, percentile:pct };
    });
  },

  async ask(question){
    if(this.busy) return;
    this.busy=true;
    document.getElementById("tutorAsk").disabled=true;
    document.getElementById("tutorExplain").disabled=true;
    const out=document.getElementById("tutorOut");
    const note=document.getElementById("tutorNote"); note.textContent=""; note.className="tutornote";
    out.replaceChildren();
    const textEl=document.createElement("span"); const cur=document.createElement("span"); cur.className="cursor";
    out.appendChild(textEl); out.appendChild(cur);

    let acc="";
    try{
      const payload = {
        prob: +this.app.prob.toFixed(3),
        verdict: this.app.prob>=0.5 ? "likely a real planet" : "likely a false positive",
        name: this.app.cur ? this.app.cur.name : null,
        nasa_disposition: this.app.cur ? this.app.cur.disposition : null,
        features: this.features(),
        question: question || ""
      };
      const res=await fetch("/api/tutor",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
      if(!res.ok || !res.body) throw new Error("no stream");
      const reader=res.body.getReader(), dec=new TextDecoder(); let buf="";
      for(;;){
        const {value,done}=await reader.read(); if(done) break;
        buf+=dec.decode(value,{stream:true}); let nl;
        while((nl=buf.indexOf("\n\n"))>=0){
          const line=buf.slice(0,nl); buf=buf.slice(nl+2);
          if(!line.startsWith("data:")) continue;
          const ev=JSON.parse(line.slice(5).trim());
          if(ev.error) throw new Error(ev.error);
          if(ev.text){ acc+=ev.text; textEl.textContent=acc; }
        }
      }
      cur.remove();
      if(!acc.trim()){ note.className="tutornote err"; note.textContent="The tutor returned an empty response."; }
    }catch(e){
      cur.remove();
      const msg=(e&&e.message)||String(e);
      if(msg.includes("Failed to fetch")||msg.includes("NetworkError")||msg==="no stream"){
        note.className="tutornote err";
        note.replaceChildren(document.createTextNode("Tutor server not reachable. Run "),
          code("export OPENAI_API_KEY=sk-..."), document.createTextNode(" then "),
          code("python server/server.py"), document.createTextNode(" (or open the live demo)."));
      } else { note.className="tutornote err"; note.textContent="Tutor error: "+msg; }
    }finally{
      this.busy=false;
      document.getElementById("tutorAsk").disabled=false;
      document.getElementById("tutorExplain").disabled=false;
    }
  }
};
function code(t){ const c=document.createElement("code"); c.textContent=t; return c; }
function setLoad(m){ const el=document.getElementById("loadmsg"); if(el) el.textContent=m; }
window.addEventListener("load", ()=>App.init());
