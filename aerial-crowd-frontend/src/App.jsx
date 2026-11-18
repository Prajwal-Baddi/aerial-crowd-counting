import {useEffect,useRef,useState} from "react";
import React from "react";
import {Activity,ArrowUpRight,Check,Clock3,CloudUpload,Image as ImageIcon,LoaderCircle,RotateCcw,ScanLine,ShieldCheck,Users,X} from "lucide-react";
const API_URL=import.meta.env.VITE_API_URL||"http://localhost:8000";
const TYPES=["image/jpeg","image/png","image/webp"];
export default function App(){
 const input=useRef(null),[file,setFile]=useState(null),[preview,setPreview]=useState(""),[result,setResult]=useState(null),[status,setStatus]=useState("idle"),[error,setError]=useState(""),[drag,setDrag]=useState(false);
 useEffect(()=>{if(!file){setPreview("");return}const u=URL.createObjectURL(file);setPreview(u);return()=>URL.revokeObjectURL(u)},[file]);
 function choose(f){setError("");setResult(null);setStatus("idle");if(!f)return;if(!TYPES.includes(f.type)){setError("Choose a JPG, PNG, or WebP image.");return}if(f.size>20*1024*1024){setError("Image must be smaller than 20 MB.");return}setFile(f)}
 function reset(){setFile(null);setResult(null);setError("");setStatus("idle");if(input.current)input.current.value=""}
 async function analyze(){if(!file)return;setStatus("loading");setError("");try{const fd=new FormData();fd.append("file",file);const r=await fetch(`${API_URL}/predict`,{method:"POST",body:fd});const data=await r.json().catch(()=>({}));if(!r.ok)throw Error(data.detail||data.error||`Request failed (${r.status})`);const count=Number(data.predicted_count??data.count);if(!Number.isFinite(count)||count<0)throw Error("API response must include a valid predicted_count.");setResult({count:Math.round(count),time:data.inference_time_ms,map:data.density_map_url});setStatus("success")}catch(e){setError(e.message==="Failed to fetch"?`Cannot connect to ${API_URL}. Start the FastAPI backend and retry.`:e.message);setStatus("error")}}
 return <main className="shell">
  <header className="topbar"><a className="brand" href="#"><span className="brandmark"><ScanLine size={21}/></span>aircount<span className="blue">.</span></a><div className="topmeta"><i/> Analysis workspace <b/> <strong>COMPUTER VISION</strong></div></header>
  <section className="heading"><div><div className="eyebrow"><i/> AERIAL IMAGE ANALYSIS</div><h1>Pedestrian <span>Counting</span></h1><p>Upload a drone image to estimate the number of pedestrians in the scene.</p></div><div className="privacy"><ShieldCheck size={16}/> Images processed by your model API</div></section>
  <section className="grid">
   <article className="panel"><div className="panelhead"><div><small>STEP 01</small><h2>Upload image</h2></div><span className="panelicon"><CloudUpload size={19}/></span></div>
    {!file?<div className={`dropzone ${drag?"drag":""}`} onClick={()=>input.current?.click()} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);choose(e.dataTransfer.files?.[0])}} role="button" tabIndex="0" onKeyDown={e=>e.key==="Enter"&&input.current?.click()}>
      <input ref={input} hidden type="file" accept="image/jpeg,image/png,image/webp" onChange={e=>choose(e.target.files?.[0])}/>
      <div className="uploadicon"><CloudUpload size={27}/></div><h3>Drop your drone image here</h3><p>or <span>browse files</span> from your device</p><div className="formats"><b>JPG</b><b>PNG</b><b>WEBP</b><i/> Max 20 MB</div>
     </div>:<div className="selected"><div className="preview"><img src={preview} alt="Uploaded drone image"/><label><ImageIcon size={13}/> IMAGE PREVIEW</label></div><div className="fileline"><span className="fileicon"><ImageIcon size={18}/></span><div><strong>{file.name}</strong><small>{(file.size/1048576).toFixed(2)} MB · Ready for analysis</small></div><button className="iconbtn" onClick={reset} aria-label="Remove image"><X size={17}/></button></div></div>}
    {error&&<div className="error"><X size={15}/>{error}</div>}
    <div className="uploadfoot"><span><ShieldCheck size={14}/> JPG, PNG or WebP</span>{file&&<button onClick={reset}><RotateCcw size={13}/> Change image</button>}</div>
    <button className="analyze" disabled={!file||status==="loading"} onClick={analyze}>{status==="loading"?<><LoaderCircle className="spin" size={18}/> Analyzing image...</>:<>Analyze image <ArrowUpRight size={18}/></>}</button><p className="hint">Your image will be processed by the pedestrian counting model.</p>
   </article>
   <article className="panel resultpanel"><div className="panelhead"><div><small>STEP 02</small><h2>Analysis result</h2></div><span className="panelicon"><Activity size={19}/></span></div>
    {status==="success"&&result?<div className="result"><div className="complete"><Check size={14}/> ANALYSIS COMPLETE</div><div className="countbox"><div><Users size={17}/> ESTIMATED PEDESTRIANS</div><strong>{result.count.toLocaleString("en-IN")}</strong><p>People estimated in the uploaded aerial image</p></div>
      {result.map&&<div className="density"><b>Density map</b><img src={result.map.startsWith("http")?result.map:`${API_URL}${result.map}`} alt="Predicted density map"/></div>}
      <div className="metadata"><div><small>File analyzed</small><b>{file?.name}</b></div><div><small>Inference time</small><b>{result.time!=null?`${Number(result.time).toFixed(1)} ms`:"Not provided"}</b></div></div><button className="secondary" onClick={reset}><RotateCcw size={15}/> Analyze another image</button>
     </div>:status==="loading"?<div className="empty loading"><div className="loader"><LoaderCircle className="spin" size={29}/></div><h3>Analyzing your image</h3><p>The model is processing the image and estimating pedestrian density. This may take a few moments.</p><div className="progress"><i/></div></div>:<div className="empty"><div className="art"><div className="sky">{[1,2,3,4,5].map(x=><i key={x}/>)}</div><div className="ground"/><div className="scan"><ScanLine size={27}/></div></div><h3>Waiting for an image</h3><p>Your pedestrian count will appear here after you upload and analyze a drone image.</p><div className="placeholders"><div><Users size={16}/> Pedestrian count <b>—</b></div><div><Clock3 size={16}/> Inference time <b>—</b></div></div></div>}
   </article>
  </section><footer><span>aircount<span className="blue">.</span> <small>Aerial pedestrian analysis</small></span><small>Powered by deep learning</small></footer>
 </main>
}
