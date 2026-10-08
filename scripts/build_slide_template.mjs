// Design-maintenance only: uses the ChatGPT bundled Artifact Tool runtime.
// Normal free CPU/Colab runs fill the committed template with Python stdlib.
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const root=process.env.CRM_REPO_ROOT;
const build=process.env.CRM_DECK_BUILD;
if (!root || !build) throw Error('Set CRM_REPO_ROOT and CRM_DECK_BUILD absolute paths');
const skill='/root/.codex/skills/builtins/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const content=JSON.parse(await fs.readFile(path.join(root,'reports/crm/final/deliverables/slide_content.json'),'utf8'));
const manifest=JSON.parse(await fs.readFile(path.join(root,'reports/crm/final/run_manifest.json'),'utf8'));
const team=['Hong Thai Phan','Nguyen Khanh An Tran','Hoang Thien Bao Bui'];
await fs.mkdir(build,{recursive:true});
await fs.mkdir(path.join(root,'assets'),{recursive:true});
function create(template) {
  const deck=Presentation.create({slideSize:{width:1280,height:720}});
  const token=(name,value)=>template?`[[${name}]]`:value;
  function text(sl,name,value,x,y,w,h,size=30,color='#122A3A',bold=false){
    const box=sl.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    box.text=token(name,value);
    box.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none'};
  }
  for(const [index,item] of content.entries()){
    const n=index+1; const sl=deck.slides.add(); sl.background.fill='#F4F7F8';
    text(sl,`kicker${n}`,item.kicker,64,30,1152,30,16,'#127D88',true);
    text(sl,`title${n}`,item.title,64,85,1152,110,42,'#122A3A',true);
    if(item.table){
      const values=item.table.map((row,r)=>row.map((v,c)=>token(`cell${n}_${r}_${c}`,v)));
      const table=sl.tables.add({rows:values.length,columns:4,left:64,top:205,width:1152,height:322,
        columnWidths:[420,242,242,248],values});
      table.borders.assign({style:'solid',fill:'#FFFFFF',width:1});
      for(let r=0;r<values.length;r++) for(let c=0;c<4;c++){
        const cell=table.getCell(r,c);
        cell.fill=r===0?'#127D88':(r%2?'#E4EEF0':'#FFFFFF');
        cell.text.style={typeface:'Arial',fontSize:24,bold:r===0,color:r===0?'#FFFFFF':'#122A3A'};
      }
      item.lines.forEach((line,j)=>text(sl,`line${n}_${j}`,line,64,555+j*47,1152,43,26));
    } else {
      item.lines.forEach((line,j)=>text(sl,`line${n}_${j}`,line,76,208+j*95,1132,86,index===0?34:30));
    }
    text(sl,`footer${n}`,`CS 582 Group 2   ${manifest.mode}   ${String(n).padStart(2,'0')}/12`,64,676,1152,24,14,'#506574');
    sl.speakerNotes.textFrame.setText(token(`note${n}`,`Suggested speaker: ${team[item.speaker_index]} (${item.seconds} seconds).\n\n${item.note}\n\nEvidence: ${item.source}`));
  }
  return deck;
}
const deck=create(false);
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
await fs.mkdir(path.join(build,'output'),{recursive:true});
const final=path.join(build,'output/validated.pptx');
await finalizePresentation({workspaceDir:build,candidatePath:candidate,finalPath:final,
  explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[6],
  pythonExecutable:process.env.CODEX_PRIMARY_RUNTIME_PYTHON,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','6'],
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'validation.json')});
await fs.copyFile(final,path.join(root,'reports/crm/final/deliverables/CRM_Final_Presentation.pptx'));
await (await PresentationFile.exportPptx(create(true))).save(path.join(root,'assets/CRM_History_Agent_Template.pptx'));
for(let i=0;i<deck.slides.items.length;i++){
  const preview=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));
}
console.log('Created validated editable deck and free-runtime template');
