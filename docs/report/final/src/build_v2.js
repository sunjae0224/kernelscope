// Build the restructured final report (v2) from content_v2.json; sections flow like the midterm report.
const fs = require('fs');
const path = require('path');
const {Document,Packer,Paragraph,TextRun,ImageRun,Table,TableRow,TableCell,
  WidthType,AlignmentType,VerticalAlign,BorderStyle,Footer,PageNumber,
  HeadingLevel,HeightRule,ShadingType,LineRuleType,LevelFormat}=require('docx');
const ROOT=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'content_v2.json'),'utf8'));
const FONT={ascii:'맑은 고딕',eastAsia:'맑은 고딕',hAnsi:'맑은 고딕',cs:'맑은 고딕'};
const W=9070;
// **bold** and [[ ]] (blank to be measured, highlighted) inside text.
const runs=(text,opts={})=>String(text).split(/(\*\*[^*]+\*\*|\[\[[^\]]*\]\])/).filter(Boolean).map(s=>{
  if(s.startsWith('[['))return new TextRun({text:'[      ]',font:FONT,size:21,highlight:'yellow',...opts});
  return new TextRun({text:s.startsWith('**')?s.slice(2,-2):s,font:FONT,size:21,bold:s.startsWith('**'),...opts});});
const p=(text,opts={})=>new Paragraph({children:runs(text,opts.run||{}),alignment:opts.align||AlignmentType.JUSTIFIED,
  spacing:{after:110,line:330},widowControl:true,...opts});
const heading=(text,level,pageBreakBefore=false)=>new Paragraph({heading:[HeadingLevel.HEADING_1,HeadingLevel.HEADING_2,HeadingLevel.HEADING_3][level-1],pageBreakBefore,
  children:runs(text,{bold:true,size:[29,23,21][level-1]}),
  spacing:{before:[0,260,160][level-1],after:[160,120,90][level-1],line:330},keepNext:true});
function cell(text,width,header=false){return new TableCell({width:{size:width,type:WidthType.DXA},verticalAlign:VerticalAlign.CENTER,
  margins:{top:65,bottom:65,left:90,right:90},shading:header?{type:ShadingType.CLEAR,fill:'E7E7E7'}:undefined,
  children:String(text).split('\n').map(t=>p(t,{alignment:AlignmentType.CENTER,spacing:{after:0,line:255},run:{size:17,bold:header}}))});}
function table(b){const parts=b.widths;const sum=parts.reduce((a,c)=>a+c,0);
 const ws=parts.map(n=>Math.round(W*n/sum));ws[ws.length-1]+=W-ws.reduce((a,c)=>a+c,0);
 return [p(b.caption,{align:AlignmentType.CENTER,run:{size:18},keepNext:true,spacing:{before:130,after:70,line:280}}),
 new Table({width:{size:W,type:WidthType.DXA},columnWidths:ws,rows:[
 new TableRow({tableHeader:true,children:b.headers.map((s,i)=>cell(s,ws[i],true))}),
 ...b.rows.map(r=>new TableRow({cantSplit:true,children:r.map((s,i)=>cell(s,ws[i]))}))]}),
 p('',{spacing:{after:60,line:90}})];}
function block(b){
 if(b.type==='p')return[p(b.text,b.align==='left'?{align:AlignmentType.LEFT}:{})];
 if(b.type==='h2')return[heading(b.text,2)];
 if(b.type==='h3')return[heading(b.text,3)];
 if(b.type==='formula')return[p(b.text,{align:AlignmentType.CENTER,run:{size:20},spacing:{before:100,after:140,line:320}})];
 if(b.type==='note')return[p(b.text,{run:{size:17},spacing:{before:40,after:120,line:275}})];
 if(b.type==='bullets')return b.items.map(t=>new Paragraph({children:runs(t),numbering:{reference:'bullets',level:0},
  alignment:AlignmentType.JUSTIFIED,spacing:{after:80,line:330}}));
 if(b.type==='table')return table(b);
 if(b.type==='figure'){const file=path.resolve(ROOT,b.file);const z=fs.readFileSync(file);const iw=z.readUInt32BE(16),ih=z.readUInt32BE(20);
  const width=b.width||565;const height=Math.round(width*ih/iw);
  if(height>360)throw new Error('Figure exceeds half a page: '+b.file);
  return [new Paragraph({alignment:AlignmentType.CENTER,keepNext:true,spacing:{before:100,after:0,line:240,lineRule:LineRuleType.AUTO},children:[new ImageRun({type:'png',data:z,transformation:{width,height}})]}),
  p(b.caption,{align:AlignmentType.CENTER,run:{size:18},spacing:{after:150,line:280}})];}
 if(b.type==='code')return b.text.split('\n').map(line=>p(line||' ',{alignment:AlignmentType.LEFT,run:{size:15,font:{...FONT,ascii:'Liberation Mono',hAnsi:'Liberation Mono'}},
  shading:{type:ShadingType.CLEAR,fill:'F3F3F3'},spacing:{after:0,line:225},keepLines:true}));
 if(b.type==='ref')return[p('['+b.n+'] '+b.text,{alignment:AlignmentType.LEFT,run:{size:19},spacing:{after:110,line:290},indent:{left:450,hanging:450}})];
 throw new Error('Unknown block '+b.type);}
const center=(t,size=24,before=0,after=0)=>p(t,{align:AlignmentType.CENTER,run:{size},spacing:{before,after,line:360}});
const cover=[p('붙임3',{run:{size:20,bold:true},spacing:{after:650,line:300}}),
 center('연구논문/작품 최종보고서',40,0,230),
 new Paragraph({spacing:{after:600},border:{bottom:{style:BorderStyle.SINGLE,color:'808080',size:20}}}),
 center('2026 학년도 제 2 학기',27,0,700),
 p('제목 : '+data.title,{run:{size:28,bold:true},spacing:{after:650,line:430},alignment:AlignmentType.LEFT}),
 center('이선재 (2021313805)',26,0,120),
 center('GitHub : https://github.com/sunjae0224/kernelscope',18,0,700),
 center('2026 년      월      일',26,0,350),
 center('지도교수:  민 동 문       서명 ______________',25,0,450),
 new Table({width:{size:W,type:WidthType.DXA},columnWidths:[1512,1512,1512,1512,1512,1510],rows:[
 new TableRow({children:['계획(10)','주제(20)','개념(20)','상세(30)','보고서(20)','총점(100)'].map(t=>cell(t,1512,true))}),
 new TableRow({height:{value:750,rule:HeightRule.ATLEAST},children:Array.from({length:6},()=>cell('',1512))})]}),
 p('* 지도교수가 평가결과 기재',{run:{size:17},spacing:{before:80,after:0,line:260}})];
const footer=new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({text:'- ',font:FONT,size:18}),new TextRun({children:[PageNumber.CURRENT],font:FONT,size:18}),new TextRun({text:' -',font:FONT,size:18})]})]});
const page={size:{width:11906,height:16838},margin:{top:1418,bottom:1418,left:1418,right:1418,footer:700}};
const body=[];
// A new page starts on the heading itself, so a full previous page never spills an empty break paragraph.
for(const [i,s] of data.sections.entries()){
 if(i&&!s.newpage)body.push(p('',{spacing:{after:200,line:240}}));
 body.push(heading(s.title,1,Boolean(i&&s.newpage)));
 for(const b of s.blocks)body.push(...block(b));
}
const doc=new Document({creator:'이선재',title:data.title,
 numbering:{config:[{reference:'bullets',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,
  style:{paragraph:{indent:{left:420,hanging:280}}}}]}]},
 styles:{default:{document:{run:{font:FONT,size:21},paragraph:{spacing:{line:330}}}},paragraphStyles:[
 {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,paragraph:{outlineLevel:0}},
 {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,paragraph:{outlineLevel:1}},
 {id:'Heading3',name:'Heading 3',basedOn:'Normal',next:'Normal',quickFormat:true,paragraph:{outlineLevel:2}}]},
 sections:[{properties:{page},children:cover},{properties:{page:{...page,pageNumbers:{start:1}}},footers:{default:footer},children:body}]});
const out=process.argv[2]||path.join(ROOT,'연구논문작품_최종보고서_v2.docx');
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(out,b);console.log(out+' ('+b.length+' bytes)');});
