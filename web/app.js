const {useState,useEffect} = React

function App(){
  const [source,setSource]=useState("dataset")
  const [zones,setZones]=useState("zones.json")
  const [employeeSide,setEmployeeSide]=useState("right")
  const [frameStride,setFrameStride]=useState(2)
  const [threshold,setThreshold]=useState(0.5)
  const [handThreshold,setHandThreshold]=useState(0.4)
  const [worldModel,setWorldModel]=useState("yoloworld.pt")
  const [handModel,setHandModel]=useState("hand_yolov8.pt")
  const [forceImageio,setForceImageio]=useState(false)
  const [runId,setRunId]=useState("")
  const [runStatus,setRunStatus]=useState(null)
  const [streamSrc,setStreamSrc]=useState("")
  const [interactions,setInteractions]=useState([])
  const [events,setEvents]=useState([])
  const [zoneInfo,setZoneInfo]=useState(null)
  const [drawMode,setDrawMode]=useState("")
  const [clicks,setClicks]=useState([])

  useEffect(()=>{fetch("/health").then(r=>r.json()).then(()=>{}).catch(()=>{})},[])
  useEffect(()=>{
    if(!runId)return
    const t=setInterval(()=>{fetch(`/process/${runId}`).then(r=>r.json()).then(setRunStatus).catch(()=>{})},1500)
    return ()=>clearInterval(t)
  },[runId])

  const start=()=>{
    const body={source,zones,employee_side:employeeSide,frame_stride:Number(frameStride),threshold:Number(threshold),hand_threshold:Number(handThreshold),world_model:worldModel,hand_model:handModel,force_imageio:Boolean(forceImageio)}
    fetch("/process",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json()).then(d=>{setRunId(d.run_id);setRunStatus({status:"starting"});setStreamSrc(`/stream/${d.run_id}`)}).catch(()=>{})
  }
  const loadZones=()=>{fetch(`/zones?path=${encodeURIComponent(zones)}`).then(r=>r.json()).then(setZoneInfo)}
  const saveZonesLine=()=>{fetch(`/zones`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({employee_side:employeeSide,line_x:Math.floor(window.line_x||0),path:zones})}).then(()=>loadZones())}
  const saveZonesSegment=()=>{const p1=window.p1||[0,0];const p2=window.p2||[0,0];fetch(`/zones`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({employee_side:employeeSide,p1,p2,path:zones})}).then(()=>loadZones())}
  const loadInteractions=()=>{fetch(`/interactions`).then(r=>r.json()).then(d=>setInteractions(d.rows||[]))}
  const loadEvents=()=>{fetch(`/events`).then(r=>r.json()).then(d=>setEvents(d.rows||[]))}
  const onImgClick=(e)=>{
    if(!drawMode)return
    const img=e.target
    const rect=img.getBoundingClientRect()
    const rx=(e.clientX-rect.left)/rect.width
    const ry=(e.clientY-rect.top)/rect.height
    const x=Math.floor(rx*640)
    const y=Math.floor(ry*640)
    const pts=[...clicks,[x,y]]
    setClicks(pts)
    if(drawMode==='line'&&pts.length>=2){
      fetch(`/zones/clear`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path:zones,clear:'all'})})
        .then(()=>{
          const p1=pts[pts.length-2]
          const p2=pts[pts.length-1]
          fetch(`/zones`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({employee_side:employeeSide,p1,p2,path:zones})}).then(()=>{setDrawMode("");setClicks([]);loadZones()})
        })
    }
    if(drawMode==='emp'&&pts.length>=2){
      fetch(`/zones/clear`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path:zones,clear:'emp'})})
        .then(()=>{
          const a=pts[pts.length-2]
          const b=pts[pts.length-1]
          const rect=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])]
          fetch(`/zones`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({employee_side:employeeSide,employee_zone:rect,path:zones})}).then(()=>{setDrawMode("");setClicks([]);loadZones()})
        })
    }
    if(drawMode==='vis'&&pts.length>=2){
      fetch(`/zones/clear`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path:zones,clear:'vis'})})
        .then(()=>{
          const a=pts[pts.length-2]
          const b=pts[pts.length-1]
          const rect=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])]
          fetch(`/zones`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({employee_side:employeeSide,visitor_zone:rect,path:zones})}).then(()=>{setDrawMode("");setClicks([]);loadZones()})
        })
    }
  }

  return React.createElement("div",{className:"wrap"},
    React.createElement("h1",null,"Detection App UI"),
    React.createElement("div",{className:"card"},
      streamSrc?React.createElement("img",{src:streamSrc,onClick:onImgClick,style:{maxWidth:"100%",border:"1px solid #45a29e",borderRadius:"6px"}}):React.createElement("div",null,"Click Start to begin"),
    ),
    React.createElement("div",{className:"card"},
      React.createElement("div",{className:"grid"},
        React.createElement("div",null,
          React.createElement("label",null,"Source"),
          React.createElement("input",{value:source,onChange:e=>setSource(e.target.value)})
        ),
        React.createElement("div",null,
          React.createElement("label",null,"Zones file"),
          React.createElement("input",{value:zones,onChange:e=>setZones(e.target.value)})
        ),
        React.createElement("div",null,
          React.createElement("label",null,"Employee side"),
          React.createElement("select",{value:employeeSide,onChange:e=>setEmployeeSide(e.target.value)},
            React.createElement("option",{value:"right"},"right"),
            React.createElement("option",{value:"left"},"left")
          )
        ),
        React.createElement("div",null,
          React.createElement("label",null,"Frame stride"),
          React.createElement("input",{type:"number",value:frameStride,onChange:e=>setFrameStride(e.target.value)})
        ),
        React.createElement("div",null,
          React.createElement("label",null,"Threshold"),
          React.createElement("input",{type:"number",step:"0.05",value:threshold,onChange:e=>setThreshold(e.target.value)})
        ),
        React.createElement("div",null,
          React.createElement("label",null,"Hand threshold"),
          React.createElement("input",{type:"number",step:"0.05",value:handThreshold,onChange:e=>setHandThreshold(e.target.value)})
        ),
        React.createElement("div",null,
          React.createElement("label",null,"World model"),
          React.createElement("input",{value:worldModel,onChange:e=>setWorldModel(e.target.value)})
        ),
        React.createElement("div",null,
          React.createElement("label",null,"Hand model"),
          React.createElement("input",{value:handModel,onChange:e=>setHandModel(e.target.value)})
        ),
        React.createElement("div",null,
          React.createElement("label",null,"Force imageio"),
          React.createElement("input",{type:"checkbox",checked:forceImageio,onChange:e=>setForceImageio(e.target.checked)})
        )
      ),
      React.createElement("div",{className:"row"},
        React.createElement("div",null,React.createElement("button",{onClick:start},"Start")),
        React.createElement("div",null,React.createElement("button",{onClick:()=>setDrawMode('line')},"Add Line Segment")),
        React.createElement("div",null,React.createElement("button",{onClick:()=>setDrawMode('emp')},"Add Employee Zone")),
        React.createElement("div",null,React.createElement("button",{onClick:()=>setDrawMode('vis')},"Add Visitors Zone")),
        React.createElement("div",null,React.createElement("button",{onClick:loadZones},"Load Zones"))
      ),
      React.createElement("div",{className:"status"},
        React.createElement("div",{className:"pill"},`Run: ${runId||'-'}`),
        React.createElement("div",{className:"pill"},`Status: ${(runStatus&&runStatus.status)||'-'}`),
        React.createElement("div",{className:"pill"},`Frames: ${(runStatus&&runStatus.frames)||0}`),
        React.createElement("div",{className:"pill"},`Customers: ${(runStatus&&runStatus.customers)||0}`)
      ),
      React.createElement("div",{style:{marginTop:12}},
        React.createElement("pre",null,zoneInfo?JSON.stringify(zoneInfo,null,2):"Zones not loaded")
      )
    ),
    React.createElement("div",{className:"card"},
      React.createElement("div",{className:"row"},
        React.createElement("div",null,React.createElement("button",{onClick:loadInteractions},"Load Interactions")),
        React.createElement("div",null,React.createElement("button",{onClick:loadEvents},"Load Events"))
      ),
      React.createElement("div",{className:"row"},
        React.createElement("div",{className:"list"},
          interactions.map(i=>React.createElement("div",{key:i.interaction_id,className:"item"},
            React.createElement("div",null,`#${i.interaction_id} ${i.classification} ${i.cnic_detected?'CNIC':''}`),
            React.createElement("div",null,`${i.timestamp_start} -> ${i.timestamp_end||'-'}`),
            React.createElement("div",null,`${i.cnic_image_path||''}`)
          ))
        ),
        React.createElement("div",{className:"list"},
          events.map(e=>React.createElement("div",{key:e.event_id,className:"item"},
            React.createElement("div",null,`#${e.event_id} ${e.event_type} ${e.zero_shot_result}`),
            React.createElement("div",null,`interaction ${e.interaction_id}`),
            React.createElement("div",null,`${e.timestamp}`),
            React.createElement("div",null,`${e.image_path}`)
          ))
        )
      )
    )
  )
}

ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App))