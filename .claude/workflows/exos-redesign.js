export const meta = {
  name: 'exos-redesign',
  description: 'Generate diverse premium art directions for the Exos exoplanet app, judge them, and synthesize one locked, implementable design blueprint',
  phases: [
    { title: 'Directions', detail: '3 distinct premium art directions in parallel' },
    { title: 'Synthesize', detail: 'judge + merge into one implementable blueprint' },
  ],
}

const PRODUCT = `Exos — a web app for a STEM-education hackathon (judges = the audience). It uses NASA Kepler+TESS data: a machine-learning model runs in the browser (ONNX) and predicts whether a star's transit signal is a REAL exoplanet or a FALSE POSITIVE. Features on one long scrolling page: a header, a hero, an input card (preset "chips" + a Simple/All toggle + range sliders), a "model's call" card (big verdict text, a confidence bar, three numeric readouts, an animated transit light-curve <canvas>, derived science tags), a "why" feature-contribution bar panel, a Missions learning game (4 cards that complete when you hit a condition), an AI tutor chat card, two explainer sections, a metrics row, and a footer. Vanilla HTML/CSS/JS, single self-contained app.css. Audience: students + technical hackathon judges.`

const GOAL = `Make it look ASTONISHING, premium, intentional, and SCIENTIFIC/instrument-grade — the opposite of generic AI "vibecoded" output. HARD BANS: the font Inter (and Roboto/Arial/system), purple-on-dark gradients, emoji used as UI chrome, oversized bubbly border-radii, flat evenly-distributed candy palettes, generic glowy gradient buttons. Embrace: a DISTINCTIVE display typeface + a refined body face + a MONOSPACE face for all numeric/data readouts (this is a scientific instrument); a cohesive palette with ONE confident signature accent plus semantic planet/false-positive colors; real depth and atmosphere (fine grain/noise, hairline 1px rules, faint grid/tick marks, restrained glow, layered transparency); tighter intentional radii; crisp inline-SVG or CSS icons instead of emoji; one orchestrated staggered page-load reveal. Do NOT default to Space Grotesk — pick something more characterful and justify it.`

const CONTRACT = `It must remain a single app.css restyle + minimal <head>/icon HTML edits. PRESERVE every element id and class name (JS depends on them): ids stars,loading,loadmsg,bN,bAUC,mAcc,mAUC,mN,mModel,cite,repolink,chips,modetoggle,sliders,verdict,conffill,confpct,probv,radv,depthv,derived,transit,vizcap,truth,bars,missions,mprog,tutorQ,tutorAsk,tutorExplain,qchips,tutorOut,tutorNote; classes wrap,card,pad,hero,badges,pill,grid,logo,orb,btnlink,chip(+confirmed/false_positive/candidate/dotc),slider(+adv/top/lab/val),modetoggle(button.on),verdict,confbar,conffill,confrow,nums,num,derived,tagd(+hz-yes/hz-hot/hz-cold),viz,vizcap,truth,bars,bar(+barwrap/barmid/barfill/barval),legend,dot,missions,mission(+solved/mh/mstate/mgoal/mteach),btn(+ghost),inputrow,tutorout(+cursor),tutornote(+err),qchips,qchip,block,lead,lessons,lesson,metrics,metric,footer,cite,skip,loading,spinner. The transit <canvas> is drawn by JS; the blueprint must specify exact hex for star, planet light-curve line, false-positive line, and marker so the canvas can be recolored to match.`

const DIRECTION_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['name','concept','fonts','palette','radii','borders_shadows','layout','motion','icons','texture','signature','anti_slop'],
  properties:{
    name:{type:'string'},
    concept:{type:'string', description:'1-2 sentences: the art direction and why it fits a NASA-data science instrument'},
    fonts:{type:'object', additionalProperties:false, required:['display','body','mono','google_href'],
      properties:{ display:{type:'string'}, body:{type:'string'}, mono:{type:'string'},
        google_href:{type:'string', description:'a real Google Fonts css2 href loading all three families'} }},
    palette:{type:'object', additionalProperties:false, required:['bg','surface','surface2','ink','muted','line','accent','planet','falsepos','notes'],
      properties:{ bg:{type:'string'},surface:{type:'string'},surface2:{type:'string'},ink:{type:'string'},muted:{type:'string'},line:{type:'string'},accent:{type:'string'},planet:{type:'string'},falsepos:{type:'string'},notes:{type:'string'} }},
    radii:{type:'string'}, borders_shadows:{type:'string'},
    layout:{type:'string'}, motion:{type:'string'}, icons:{type:'string'}, texture:{type:'string'},
    signature:{type:'string', description:'the ONE unforgettable detail'},
    anti_slop:{type:'string', description:'how this specifically avoids the AI-default look'},
  }
}

const ANGLES = [
  { key:'observatory-terminal', brief:'Angle A — "Observatory data terminal / instrument readout": precise, engineered, monospace-forward, hairline grids and tick marks, telemetry feel. Restrained, confident, near-black. Think a professional telescope control surface.' },
  { key:'editorial-cosmos', brief:'Angle B — "Editorial science magazine" (Quanta / NASA feature-article energy): a high-contrast display SERIF for headlines as the hero move, generous negative space, refined and sophisticated, color used sparingly as ink-on-deep-field. Unexpected for a data tool, very designed.' },
  { key:'mission-control', brief:'Angle C — "Mission control / JPL operations": amber-and-signal palette on black, dense structured telemetry grid, label/value pairs, status-light semantics, a built, utilitarian-but-beautiful console aesthetic.' },
]

phase('Directions')
const directions = (await parallel(ANGLES.map(a => () =>
  agent(
    `You are a world-class product/visual designer. Design ONE distinctive art direction for this app.\n\nPRODUCT:\n${PRODUCT}\n\nGOAL:\n${GOAL}\n\nCONSTRAINTS:\n${CONTRACT}\n\nYOUR ASSIGNED ANGLE (commit fully to it, make it astonishing and unmistakably intentional):\n${a.brief}\n\nReturn a complete, concrete spec: real font families (with a working Google Fonts href), exact hex values, radii, border/shadow language, layout/composition ideas, one orchestrated load motion, icon approach (no emoji), texture/atmosphere, the single signature detail, and how it dodges the generic AI look. Be specific and buildable.`,
    { label:`dir:${a.key}`, phase:'Directions', schema: DIRECTION_SCHEMA }
  )
))).filter(Boolean)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['chosen','rationale','fonts','css_vars','type_scale','components','motion','icon_plan','canvas_colors','anti_slop_checklist'],
  properties:{
    chosen:{type:'string', description:'which direction won (or the named hybrid)'},
    rationale:{type:'string'},
    fonts:{type:'object', additionalProperties:false, required:['display','body','mono','google_href'],
      properties:{ display:{type:'string'}, body:{type:'string'}, mono:{type:'string'}, google_href:{type:'string'} }},
    css_vars:{type:'object', additionalProperties:false, required:['bg','surface','surface2','ink','muted','line','accent','accent_soft','planet','falsepos','warn','gold','radius','shadow'],
      properties:{ bg:{type:'string'},surface:{type:'string'},surface2:{type:'string'},ink:{type:'string'},muted:{type:'string'},line:{type:'string'},accent:{type:'string'},accent_soft:{type:'string'},planet:{type:'string'},falsepos:{type:'string'},warn:{type:'string'},gold:{type:'string'},radius:{type:'string'},shadow:{type:'string'} }},
    type_scale:{type:'string', description:'sizes/weights/letter-spacing for h1, h2, section h3 labels, body, mono readouts, micro-labels'},
    components:{type:'object', additionalProperties:false,
      required:['background_texture','header','hero_and_badges','cards_and_section_headings','chips_and_modetoggle','sliders','verdict_and_confbar','nums_and_derived','transit_viz','why_bars','missions','tutor','buttons_and_inputs','lessons_and_metrics','footer_and_loading'],
      properties:{
        background_texture:{type:'string'}, header:{type:'string'}, hero_and_badges:{type:'string'},
        cards_and_section_headings:{type:'string'}, chips_and_modetoggle:{type:'string'}, sliders:{type:'string'},
        verdict_and_confbar:{type:'string'}, nums_and_derived:{type:'string'}, transit_viz:{type:'string'},
        why_bars:{type:'string'}, missions:{type:'string'}, tutor:{type:'string'}, buttons_and_inputs:{type:'string'},
        lessons_and_metrics:{type:'string'}, footer_and_loading:{type:'string'},
      }},
    motion:{type:'string', description:'the orchestrated page-load reveal + key micro-interactions, CSS-only'},
    icon_plan:{type:'string', description:'what replaces each emoji (inline SVG/CSS), described concretely'},
    canvas_colors:{type:'object', additionalProperties:false, required:['star','planet_line','falsepos_line','marker','planet_disc'],
      properties:{ star:{type:'string'}, planet_line:{type:'string'}, falsepos_line:{type:'string'}, marker:{type:'string'}, planet_disc:{type:'string'} }},
    anti_slop_checklist:{type:'array', items:{type:'string'}},
  }
}

const blueprint = await agent(
  `You are the design director. Three art directions were produced for the Exos app (below). Judge them on: astonishing/memorable, distinct-from-generic-AI, professional/credible to technical judges, fit for a NASA-data science instrument, and buildable in pure CSS while preserving the markup contract. Pick the strongest as the base and graft the best ideas from the others into ONE cohesive, locked, implementable blueprint.\n\nPRODUCT:\n${PRODUCT}\n\nGOAL:\n${GOAL}\n\nCONSTRAINTS:\n${CONTRACT}\n\nTHREE DIRECTIONS:\n${JSON.stringify(directions, null, 2)}\n\nReturn the final blueprint: chosen direction + rationale, the font trio with a real Google Fonts href, exact :root css var hex/values, a type scale, concrete styling directives for EVERY listed component group, the orchestrated CSS-only load motion, a concrete icon plan replacing all emoji, the exact canvas hex colors, and an anti-slop checklist. Be specific enough to implement directly.`,
  { label:'synthesize', phase:'Synthesize', schema: SYNTH_SCHEMA }
)

return { directions: directions.map(d => d && d.name), blueprint }
