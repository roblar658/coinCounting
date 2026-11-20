"""
Standalone generator for ./solution.py ( Coin Counting).

Reads Base85 weights from ./trained_weights.b85 (or accepts a weight string
directly via generate_solution(weights_b85=...)), writes ./solution.py, verifies
lossless weight loading & CoinDetector initialization, and optionally benchmarks
against ./sampleimages/*.in.

Usage:
    python generate_solution.py
    python generate_solution.py --test
    python generate_solution.py --weights trained_weights.b85 --out solution.py --test
"""

import argparse
import base64
import glob
import importlib.util
import math
import os
import struct
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(BASE_DIR, "trained_weights.b85")
DEFAULT_SOLUTION_PATH = os.path.join(BASE_DIR, "solution.py")
DEFAULT_SAMPLE_DIR = os.path.join(BASE_DIR, "sampleimages")
SOFT_LIMIT_BYTES = 90112

SOLUTION_TEMPLATE = r'''import base64;import gc;import math;import struct;import sys;import time;from operator import mul;_T0=time.perf_counter();WEIGHTS_B85="__WEIGHTS_B85__";_ZZ=(0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63,);_ZZ4=tuple(((nat>>3)<<2)|(nat&3)if ((nat>>3)<4 and (nat&7)<4)else -1 for nat in _ZZ);_NAT4=(0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27);_C1=math.cos(1.0*math.pi/16.0);_C2=math.cos(2.0*math.pi/16.0);_C3=math.cos(3.0*math.pi/16.0);_C4=math.cos(4.0*math.pi/16.0);_C5=math.cos(5.0*math.pi/16.0);_C6=math.cos(6.0*math.pi/16.0);_C7=math.cos(7.0*math.pi/16.0);_S0=_C4*0.5;_S1=_C1*0.5;_S2=_C2*0.5;_S3=_C3*0.5;_S5=_C5*0.5;_S6=_C6*0.5;_S7=_C7*0.5;_C8_0=0.3535533905932738;_C8_K=0.5;_H4=tuple(tuple((_C8_0 if u==0 else _C8_K*0.5*(math.cos((4*y+1)*u*math.pi/16.0)+math.cos((4*y+3)*u*math.pi/16.0)))for u in range(4))for y in range(4));_H4_C0=_H4[0][0];_H4_C2=_H4[0][2];_H4_01=_H4[0][1];_H4_03=_H4[0][3];_H4_11=_H4[1][1];_H4_13=_H4[1][3];_FILL4=tuple(bytes([v])*4 for v in range(256));_FILL8=tuple(bytes([v])*8 for v in range(256));_BIT_MASK=tuple((1<<i)-1 for i in range(17));_BIT_HALF=tuple(1<<(i-1)if i>0 else 0 for i in range(17));_CLAMP_TUP=tuple(range(256))+(255,)*4096+(0,)*4096;_ZERO16=[0.0]*16;_ZERO64=[0.0]*64;_Z8=[0.0]*8;_Y2_TBL=tuple(i*i for i in range(256));_RADII=(3.6,8.5,13.5,17.4);_ANGLES=tuple(2.0*math.pi*a/16.0 for a in range(16));_POLAR_OFF=tuple(tuple((r*math.cos(th),r*math.sin(th))for th in _ANGLES)for r in _RADII);_OUTER_OFF=tuple((20.0*math.cos(2.0*math.pi*a/12.0),20.0*math.sin(2.0*math.pi*a/12.0))for a in range(12));_COARSE_OFF=tuple(tuple((r*math.cos(2.0*math.pi*a/n_a),r*math.sin(2.0*math.pi*a/n_a))for a in range(n_a))for r,n_a in ((3.5,6),(8.5,8),(13.5,10),(17.5,10)));_COARSE_IOFF=tuple(tuple(int(math.floor(dy+0.5))*500+int(math.floor(dx+0.5))for dx,dy in ring)for ring in _COARSE_OFF);_OUTER_IOFF=tuple(int(math.floor(dy+0.5))*500+int(math.floor(dx+0.5))for dx,dy in _OUTER_OFF)
def _build_dc12(counts,syms):
 lut=[-999]*4096;slow={};bm=_BIT_MASK;bh=_BIT_HALF;code=0;idx=0
 for blen in range(1,17):
  for _ in range(counts[blen-1]):
   sz=syms[idx];idx+=1
   if blen<=12:
    tot=blen+sz
    if tot<=12:
     sh=12-tot;base=code<<(sz+sh);nrep=1<<sh
     for bits in range(1<<sz):cv=bits-bm[sz]if bits<bh[sz]else bits;pv=((cv+4096)<<5)|tot;st=base+(bits<<sh);lut[st:st+nrep]=[pv]*nrep
    else:sh=12-blen;base=code<<sh;nrep=1<<sh;lut[base:base+nrep]=[-((sz<<5)|blen)]*nrep
   else:slow[(blen,code)]=sz
   code+=1
  code<<=1
 return lut,slow
def _build_ac13(counts,syms):
 lut=[-999]*8192;slow={};bm=_BIT_MASK;bh=_BIT_HALF;code=0;idx=0
 for blen in range(1,17):
  for _ in range(counts[blen-1]):
   sym=syms[idx];idx+=1;rle=sym>>4;sz=sym&0xF
   if blen<=13:
    tot=blen+sz
    if sz==0:sh=13-blen;base=code<<sh;nrep=1<<sh;pv=blen if rle==0 else (32+blen);lut[base:base+nrep]=[pv]*nrep
    elif tot<=13:
     sh=13-tot;base=code<<(sz+sh);nrep=1<<sh
     for bits in range(1<<sz):cv=bits-bm[sz]if bits<bh[sz]else bits;pv=((cv+4096)<<9)|(rle<<5)|tot;st=base+(bits<<sh);lut[st:st+nrep]=[pv]*nrep
    else:sh=13-blen;base=code<<sh;nrep=1<<sh;lut[base:base+nrep]=[-((sym<<5)|blen)]*nrep
   else:slow[(blen,code)]=sym
   code+=1
  code<<=1
 return lut,slow
def _build_ac13_eob(ac13_lut):
 skip=[0]*8192
 for idx in range(8192):
  rem=13;cur=idx;dk=0;used=0
  while True:
   val=ac13_lut[(cur<<(13-rem))&0x1FFF]
   if val>64:
    tot=val&31
    if tot>rem:break
    step_k=((val>>5)&15)+1
    if dk+step_k>16:break
    dk+=step_k;used+=tot;rem-=tot;cur&=(1<<rem)-1
    if rem<2:break
   elif val>0:
    if val<32:
     if val<=rem:skip[idx]=used+val
     break
    blen=val-32
    if blen>rem or dk+16>16:break
    dk+=16;used+=blen;rem-=blen;cur&=(1<<rem)-1
    if rem<2:break
   else:break
 return skip
def _huff_slow(words,wpos,bitbuf,bitcnt,tbl,min_l):
 code=0;sym=0
 for blen in range(1,17):
  if bitcnt==0:bitbuf=words[wpos];wpos+=1;bitcnt=64
  bitcnt-=1;code=(code<<1)|((bitbuf>>bitcnt)&1)
  if blen>min_l and (blen,code)in tbl:sym=tbl[(blen,code)];break
 return sym,wpos,bitbuf,bitcnt
def _idct4_put(b16,pbuf,base,pw):
 _c0=_H4_C0;_c2=_H4_C2;_h01=_H4_01;_h03=_H4_03;_h11=_H4_11;_h13=_H4_13;ct=_CLAMP_TUP;pw2=pw*2;pw3=pw*3;d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15=b16;dc=d0*_c0;e2=d2*_c2;ev0=dc+e2;ev1=dc-e2;o0=d1*_h01+d3*_h03;o1=d1*_h11+d3*_h13;r00=ev0+o0;r01=ev1+o1;r02=ev1-o1;r03=ev0-o0;dc=d4*_c0;e2=d6*_c2;ev0=dc+e2;ev1=dc-e2;o0=d5*_h01+d7*_h03;o1=d5*_h11+d7*_h13;r10=ev0+o0;r11=ev1+o1;r12=ev1-o1;r13=ev0-o0;dc=d8*_c0;e2=d10*_c2;ev0=dc+e2;ev1=dc-e2;o0=d9*_h01+d11*_h03;o1=d9*_h11+d11*_h13;r20=ev0+o0;r21=ev1+o1;r22=ev1-o1;r23=ev0-o0;dc=d12*_c0;e2=d14*_c2;ev0=dc+e2;ev1=dc-e2;o0=d13*_h01+d15*_h03;o1=d13*_h11+d15*_h13;r30=ev0+o0;r31=ev1+o1;r32=ev1-o1;r33=ev0-o0
 for c,(a0,a1,a2,a3)in enumerate(((r00,r10,r20,r30),(r01,r11,r21,r31),(r02,r12,r22,r32),(r03,r13,r23,r33))):dc=a0*_c0;e2=a2*_c2;ev0=dc+e2;ev1=dc-e2;o0=a1*_h01+a3*_h03;o1=a1*_h11+a3*_h13;b=base+c;pbuf[b]=ct[int(ev0+o0+128.5)];pbuf[b+pw]=ct[int(ev1+o1+128.5)];pbuf[b+pw2]=ct[int(ev1-o1+128.5)];pbuf[b+pw3]=ct[int(ev0-o0+128.5)]
def _idct8_put(blk,nz_rows,tmp,pbuf,base,pw):
 _s0=_S0;_s1=_S1;_s2=_S2;_s3=_S3;_s5=_S5;_s6=_S6;_s7=_S7;z8=_Z8;ct=_CLAMP_TUP
 for r in range(8):
  i0=r<<3
  if (nz_rows>>r)&1:d0=blk[i0]*_s0;d2=blk[i0+2];d4=blk[i0+4]*_s0;d6=blk[i0+6];e0=d0+d4;e1=d0-d4;e2=d2*_s6-d6*_s2;e3=d2*_s2+d6*_s6;p0=e0+e3;p3=e0-e3;p1=e1+e2;p2=e1-e2;d1=blk[i0+1];d3=blk[i0+3];d5=blk[i0+5];d7=blk[i0+7];o0=d1*_s1+d3*_s3+d5*_s5+d7*_s7;o1=d1*_s3-d3*_s7-d5*_s1-d7*_s5;o2=d1*_s5-d3*_s1+d5*_s7+d7*_s3;o3=d1*_s7-d3*_s5+d5*_s3-d7*_s1;tmp[i0]=p0+o0;tmp[i0+7]=p0-o0;tmp[i0+1]=p1+o1;tmp[i0+6]=p1-o1;tmp[i0+2]=p2+o2;tmp[i0+5]=p2-o2;tmp[i0+3]=p3+o3;tmp[i0+4]=p3-o3
  else:tmp[i0:i0+8]=z8
 for c in range(8):d0=tmp[c]*_s0;d2=tmp[c+16];d4=tmp[c+32]*_s0;d6=tmp[c+48];e0=d0+d4;e1=d0-d4;e2=d2*_s6-d6*_s2;e3=d2*_s2+d6*_s6;p0=e0+e3;p3=e0-e3;p1=e1+e2;p2=e1-e2;d1=tmp[c+8];d3=tmp[c+24];d5=tmp[c+40];d7=tmp[c+56];o0=d1*_s1+d3*_s3+d5*_s5+d7*_s7;o1=d1*_s3-d3*_s7-d5*_s1-d7*_s5;o2=d1*_s5-d3*_s1+d5*_s7+d7*_s3;o3=d1*_s7-d3*_s5+d5*_s3-d7*_s1;idx=base+c;pbuf[idx]=ct[int(p0+o0+128.5)];idx+=pw;pbuf[idx]=ct[int(p1+o1+128.5)];idx+=pw;pbuf[idx]=ct[int(p2+o2+128.5)];idx+=pw;pbuf[idx]=ct[int(p3+o3+128.5)];idx+=pw;pbuf[idx]=ct[int(p3-o3+128.5)];idx+=pw;pbuf[idx]=ct[int(p2-o2+128.5)];idx+=pw;pbuf[idx]=ct[int(p1-o1+128.5)];idx+=pw;pbuf[idx]=ct[int(p0-o0+128.5)]
def _decode_jpeg_general(data,qtables,htables_raw,comps,scan_comps,dri,width,height,p,deadline=None):
 n=len(data);segments=[];chunks=[];cur=p
 while cur<n:
  ff=data.find(0xFF,cur)
  if ff==-1:chunks.append(data[cur:]);break
  nxt=data[ff+1]if ff+1<n else 0
  if nxt==0x00:chunks.append(data[cur:ff+1]);cur=ff+2
  elif nxt==0xFF:chunks.append(data[cur:ff]);cur=ff+1
  elif 0xD0<=nxt<=0xD7:
   chunks.append(data[cur:ff]);raw_s=b"".join(chunks)+b"\x00"*24;pad=(-len(raw_s))&7
   if pad:raw_s+=b"\x00"*pad
   segments.append(struct.unpack(f">{len(raw_s)//8}Q",raw_s));chunks=[];cur=ff+2
  else:chunks.append(data[cur:ff]);break
 if chunks or not segments:
  raw_s=b"".join(chunks)+b"\x00"*24;pad=(-len(raw_s))&7
  if pad:raw_s+=b"\x00"*pad
  segments.append(struct.unpack(f">{len(raw_s)//8}Q",raw_s))
 htables={}
 for (tc,th),(counts,syms)in htables_raw.items():htables[(tc,th)]=_build_dc12(counts,syms)if tc==0 else _build_ac13(counts,syms)
 max_h=max(c[1]for c in comps);max_v=max(c[2]for c in comps);mcu_cols=(width+8*max_h-1)//(8*max_h);mcu_rows=(height+8*max_v-1)//(8*max_v);hw=width>>1;hh=height>>1;can_half=(hw==500 and hh==500 and all((h==max_h and v==max_v)or (h*2==max_h and v*2==max_v)for _,h,v,_ in comps));planes=[]
 for cid,h,v,qid in comps:
  if can_half and h==max_h and v==max_v:pw=mcu_cols*h*4;ph=mcu_rows*v*4;qt=qtables[qid];qt4=tuple(qt[idx]for idx in _NAT4);planes.append([bytearray(pw*ph),pw,h,v,qt4,True])
  else:pw=mcu_cols*h*8;ph=mcu_rows*v*8;planes.append([bytearray(pw*ph),pw,h,v,qtables[qid],False])
 sched=[]
 for cid,td,ta in scan_comps:
  cidx=next(i for i,c in enumerate(comps)if c[0]==cid);dc_f,dc_s=htables[(0,td)];ac_f,ac_s=htables[(1,ta)];pbuf,pw,h,v,qt,is_half=planes[cidx];bsize=4 if is_half else 8
  for bv in range(v):
   for bh in range(h):sched.append((cidx,dc_f,dc_s,ac_f,ac_s,pbuf,pw,bsize*h,bsize*v,bh*bsize,bv*bsize,qt,is_half))
 zz=_ZZ;zz4=_ZZ4;fill4=_FILL4;fill8=_FILL8;bmask=_BIT_MASK;bhalf=_BIT_HALF;ct=_CLAMP_TUP;z16=_ZERO16;z64=_ZERO64;b16=[0.0]*16;blk=[0.0]*64;tmp=[0.0]*64;dc_pred=[0]*len(comps);seg_idx=0;words=segments[0];wpos=0;bitbuf=0;bitcnt=0;mcu_since_rst=0
 for my in range(mcu_rows):
  if deadline is not None and (my&15)==15 and time.perf_counter()>deadline:break
  for mx in range(mcu_cols):
   if dri>0 and mcu_since_rst==dri:
    mcu_since_rst=0;dc_pred=[0]*len(comps)
    if seg_idx+1<len(segments):seg_idx+=1;words=segments[seg_idx];wpos=0;bitbuf=0;bitcnt=0
    else:bitcnt&=~7
   mcu_since_rst+=1
   for cidx,dc_f,dc_s,ac_f,ac_s,pbuf,pw,sx,sy,bx0,by0,qt,is_half in sched:
    if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
    dval=dc_f[(bitbuf>>(bitcnt-12))&0xFFF]
    if dval>=0:dc_pred[cidx]+=(dval>>5)-4096;bitcnt-=(dval&31)
    else:
     if dval!=-999:neg=-dval;bitcnt-=(neg&31);sz=neg>>5
     else:sz,wpos,bitbuf,bitcnt=_huff_slow(words,wpos,bitbuf,bitcnt,dc_s,12)
     if sz:
      if bitcnt<sz:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
      bitcnt-=sz;bits=(bitbuf>>bitcnt)&bmask[sz]
      if bits<bhalf[sz]:bits-=bmask[sz]
      dc_pred[cidx]+=bits
    dc_val=dc_pred[cidx]*qt[0]
    if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
    val=ac_f[(bitbuf>>(bitcnt-13))&0x1FFF];base=(my*sy+by0)*pw+(mx*sx+bx0)
    if 0<val<32:
     bitcnt-=val;iv=ct[int(dc_val*0.125+128.5)]
     if is_half:
      rb=fill4[iv];b=base
      for _ in range(4):pbuf[b:b+4]=rb;b+=pw
     else:
      rb=fill8[iv];b=base
      for _ in range(8):pbuf[b:b+8]=rb;b+=pw
     continue
    if is_half:b16[:]=z16;b16[0]=dc_val
    else:blk[:]=z64;blk[0]=dc_val;nz_rows=1
    k=1;first=True
    while k<64:
     if not first:
      if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
      val=ac_f[(bitbuf>>(bitcnt-13))&0x1FFF]
     first=False
     if val>64:
      k+=(val>>5)&15;bitcnt-=(val&31)
      if is_half:
       if k<25:
        n4=zz4[k]
        if n4>=0:b16[n4]=((val>>9)-4096)*qt[n4]
      elif k<64:nat=zz[k];blk[nat]=((val>>9)-4096)*qt[nat];nz_rows|=1<<(nat>>3)
      k+=1
     elif val>0:
      if val<32:bitcnt-=val;break
      bitcnt-=(val-32);k+=16
     else:
      if val!=-999:neg=-val;bitcnt-=(neg&31);asym=neg>>5
      else:asym,wpos,bitbuf,bitcnt=_huff_slow(words,wpos,bitbuf,bitcnt,ac_s,13)
      if asym==0:break
      rle=asym>>4;sz=asym&0xF
      if sz==0:
       if rle==15:k+=16
       else:break
      else:
       k+=rle
       if k<64:
        if bitcnt<sz:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
        bitcnt-=sz;bits=(bitbuf>>bitcnt)&bmask[sz]
        if bits<bhalf[sz]:bits-=bmask[sz]
        if is_half:
         n4=zz4[k]
         if n4>=0:b16[n4]=bits*qt[n4]
        else:nat=zz[k];blk[nat]=bits*qt[nat];nz_rows|=1<<(nat>>3)
       k+=1
    if is_half:_idct4_put(b16,pbuf,base,pw)
    else:_idct8_put(blk,nz_rows,tmp,pbuf,base,pw)
 if can_half:
  def crop500(pl):pb=pl[0];pw=pl[1];pl[0]=b"";return bytes(pb)if (pw==500 and len(pb)==250000)else b"".join(pb[y*pw:y*pw+500]for y in range(500))
  Y=crop500(planes[0]);return (Y,crop500(planes[1]),crop500(planes[2]))if len(planes)>=3 else (Y,b"\x80"*250000,b"\x80"*250000)
 def down500(pl):
  pb,pw,h_scale,v_scale,_,_=pl;pl[0]=b"";out=bytearray(250000)
  if width==1000 and height==1000 and h_scale==max_h and v_scale==max_v:
   for y in range(500):
    r0=(y*2)*pw;r1=r0+pw;o=y*500
    for x in range(500):x2=x*2;out[o+x]=(pb[r0+x2]+pb[r0+x2+1]+pb[r1+x2]+pb[r1+x2+1]+2)>>2
  elif width==1000 and height==1000 and h_scale*2==max_h and v_scale==max_v:
   for y in range(500):
    r0=(y*2)*pw;r1=r0+pw;o=y*500
    for x in range(500):out[o+x]=(pb[r0+x]+pb[r1+x]+1)>>1
  else:
   sx=(width*h_scale/max_h)/500.0;sy=(height*v_scale/max_v)/500.0
   for y in range(500):
    ry=int(y*sy)*pw;o=y*500
    for x in range(500):out[o+x]=pb[ry+int(x*sx)]
  return bytes(out)
 Y=down500(planes[0])
 if len(planes)>=3:return Y,down500(planes[1]),down500(planes[2])
 return Y,b"\x80"*250000,b"\x80"*250000
def decode_jpeg_half(data:bytes,deadline=None):
 n=len(data);assert data[0]==0xFF and data[1]==0xD8,"Not JPEG";pos=2;qtables={};htables_raw={};comps=[];scan_comps=[];dri=0;width=height=0;p=0
 while pos<n:
  if data[pos]!=0xFF:pos+=1;continue
  while pos<n and data[pos]==0xFF:pos+=1
  if pos>=n:break
  m=data[pos];pos+=1
  if m==0xD9:break
  if m==0x00 or (0xD0<=m<=0xD7):continue
  seg_len=(data[pos]<<8)|data[pos+1];seg_end=pos+seg_len;p=pos+2
  if m==0xDB:
   while p<seg_end:
    info=data[p];p+=1;prec=info>>4;qid=info&0xF;qt=[0]*64
    for k in range(64):
     if prec==0:qt[_ZZ[k]]=data[p];p+=1
     else:qt[_ZZ[k]]=(data[p]<<8)|data[p+1];p+=2
    qtables[qid]=qt
  elif m==0xC4:
   while p<seg_end:info=data[p];p+=1;tc=info>>4;th=info&0xF;counts=data[p:p+16];p+=16;tot=sum(counts);syms=data[p:p+tot];p+=tot;htables_raw[(tc,th)]=(counts,syms)
  elif m==0xDD:dri=(data[p]<<8)|data[p+1]
  elif m in (0xC0,0xC1):
   height=(data[p+1]<<8)|data[p+2];width=(data[p+3]<<8)|data[p+4];nf=data[p+5];p+=6
   for _ in range(nf):comps.append((data[p],data[p+1]>>4,data[p+1]&0xF,data[p+2]));p+=3
  elif m==0xDA:
   ns=data[p];p+=1
   for _ in range(ns):scan_comps.append((data[p],data[p+1]>>4,data[p+1]&0xF));p+=2
   p=seg_end;break
  pos=seg_end
 if dri!=0 or width!=1000 or height!=1000 or len(comps)!=3 or len(scan_comps)!=3:return _decode_jpeg_general(data,qtables,htables_raw,comps,scan_comps,dri,width,height,p,deadline)
 (cy_id,hy,vy,qy_id),(cb_id,hcb,vcb,qcb_id),(cr_id,hcr,vcr,qcr_id)=comps
 if (hy,vy,hcb,vcb,hcr,vcr)!=(2,2,1,1,1,1)or [s[0]for s in scan_comps]!=[cy_id,cb_id,cr_id]:return _decode_jpeg_general(data,qtables,htables_raw,comps,scan_comps,dri,width,height,p,deadline)
 scan_end=data.find(b"\xff\xd9",p)
 if scan_end<0:scan_end=n
 raw_scan=data[p:scan_end]
 if b"\xff"in raw_scan:
  chunks=[];cur=0;rn=len(raw_scan)
  while cur<rn:
   ff=raw_scan.find(0xFF,cur)
   if ff==-1:chunks.append(raw_scan[cur:]);break
   nxt=raw_scan[ff+1]if ff+1<rn else 0
   if nxt==0x00:chunks.append(raw_scan[cur:ff+1]);cur=ff+2
   elif nxt==0xFF:chunks.append(raw_scan[cur:ff]);cur=ff+1
   else:chunks.append(raw_scan[cur:ff]);break
  raw_s=b"".join(chunks)+b"\x00"*24
 else:raw_s=raw_scan+b"\x00"*24
 pad=(-len(raw_s))&7
 if pad:raw_s+=b"\x00"*pad
 words=struct.unpack(f">{len(raw_s)//8}Q",raw_s);sc_map={cid:(td,ta)for cid,td,ta in scan_comps};ydc_f,ydc_s=_build_dc12(*htables_raw[(0,sc_map[cy_id][0])]);yac_f,yac_s=_build_ac13(*htables_raw[(1,sc_map[cy_id][1])]);yac_eob=_build_ac13_eob(yac_f);cbdc_f,cbdc_s=_build_dc12(*htables_raw[(0,sc_map[cb_id][0])]);cbac_f,cbac_s=_build_ac13(*htables_raw[(1,sc_map[cb_id][1])]);crdc_f,crdc_s=(cbdc_f,cbdc_s)if sc_map[cr_id][0]==sc_map[cb_id][0]else _build_dc12(*htables_raw[(0,sc_map[cr_id][0])]);crac_f,crac_s=(cbac_f,cbac_s)if sc_map[cr_id][1]==sc_map[cb_id][1]else _build_ac13(*htables_raw[(1,sc_map[cr_id][1])]);qy=qtables[qy_id];qcb=qtables[qcb_id];qcr=qtables[qcr_id];qt4_y=tuple(qy[idx]for idx in _NAT4);pw=504;pw2=1008;pw3=1512;ybuf=bytearray(pw*pw);cbbuf=bytearray(pw*pw);crbuf=bytearray(pw*pw);zz4=_ZZ4;zz8=_ZZ;bmask=_BIT_MASK;bhalf=_BIT_HALF;ct=_CLAMP_TUP;fill4=_FILL4;fill8=_FILL8;_c0=_H4_C0;_c2=_H4_C2;_h01=_H4_01;_h03=_H4_03;_h11=_H4_11;_h13=_H4_13;_s0=_S0;_s1=_S1;_s2=_S2;_s3=_S3;_s5=_S5;_s6=_S6;_s7=_S7;b16=[0.0]*16;z16=_ZERO16;blk=[0.0]*64;tmp=[0.0]*64;z64=_ZERO64;z8=_Z8;dc_y=dc_cb=dc_cr=0;bitbuf=words[0];wpos=1;bitcnt=64;qy0=qt4_y[0];qcb0=qcb[0];qcr0=qcr[0]
 for my in range(63):
  if deadline is not None and (my&15)==15 and time.perf_counter()>deadline:break
  yb_row=my*8*pw
  for mx in range(63):
   y_mx=yb_row+mx*8
   for by_off in (y_mx,y_mx+4,y_mx+2016,y_mx+2020):
    if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
    dval=ydc_f[(bitbuf>>(bitcnt-12))&0xFFF]
    if dval>=0:dc_y+=(dval>>5)-4096;bitcnt-=(dval&31)
    else:
     if dval!=-999:neg=-dval;bitcnt-=(neg&31);sz=neg>>5
     else:sz,wpos,bitbuf,bitcnt=_huff_slow(words,wpos,bitbuf,bitcnt,ydc_s,12)
     if sz:
      if bitcnt<sz:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
      bitcnt-=sz;bits=(bitbuf>>bitcnt)&bmask[sz]
      if bits<bhalf[sz]:bits-=bmask[sz]
      dc_y+=bits
    dc_val=dc_y*qy0
    if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
    val=yac_f[(bitbuf>>(bitcnt-13))&0x1FFF]
    if val>64:
     b16[:]=z16;b16[0]=dc_val;k=1+((val>>5)&15);bitcnt-=(val&31)
     if k<25:
      n4=zz4[k]
      if n4>=0:b16[n4]=((val>>9)-4096)*qt4_y[n4]
     k+=1
    elif val>0:
     if val<32:bitcnt-=val;rb=fill4[ct[int(dc_val*0.125+128.5)]];ybuf[by_off:by_off+4]=rb;ybuf[by_off+pw:by_off+pw+4]=rb;ybuf[by_off+pw2:by_off+pw2+4]=rb;ybuf[by_off+pw3:by_off+pw3+4]=rb;continue
     bitcnt-=(val-32);b16[:]=z16;b16[0]=dc_val;k=17
    else:
     if val!=-999:neg=-val;bitcnt-=(neg&31);asym=neg>>5
     else:asym,wpos,bitbuf,bitcnt=_huff_slow(words,wpos,bitbuf,bitcnt,yac_s,13)
     if asym==0:rb=fill4[ct[int(dc_val*0.125+128.5)]];ybuf[by_off:by_off+4]=rb;ybuf[by_off+pw:by_off+pw+4]=rb;ybuf[by_off+pw2:by_off+pw2+4]=rb;ybuf[by_off+pw3:by_off+pw3+4]=rb;continue
     b16[:]=z16;b16[0]=dc_val;rle=asym>>4;sz=asym&0xF
     if sz==0:k=17 if rle==15 else 64
     else:
      k=1+rle
      if k<64:
       if bitcnt<sz:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
       bitcnt-=sz;n4=zz4[k]
       if n4>=0:
        bits=(bitbuf>>bitcnt)&bmask[sz]
        if bits<bhalf[sz]:bits-=bmask[sz]
        b16[n4]=bits*qt4_y[n4]
      k+=1
    while k<25:
     if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
     val=yac_f[(bitbuf>>(bitcnt-13))&0x1FFF]
     if val>64:
      k+=(val>>5)&15;bitcnt-=(val&31)
      if k<25:
       n4=zz4[k]
       if n4>=0:b16[n4]=((val>>9)-4096)*qt4_y[n4]
      k+=1
     elif val>0:
      if val<32:bitcnt-=val;k=64;break
      bitcnt-=(val-32);k+=16
     else:
      if val!=-999:neg=-val;bitcnt-=(neg&31);asym=neg>>5
      else:asym,wpos,bitbuf,bitcnt=_huff_slow(words,wpos,bitbuf,bitcnt,yac_s,13)
      if asym==0:k=64;break
      rle=asym>>4;sz=asym&0xF
      if sz==0:
       if rle==15:k+=16
       else:k=64;break
      else:
       k+=rle
       if k<64:
        if bitcnt<sz:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
        bitcnt-=sz;n4=zz4[k]
        if n4>=0:
         bits=(bitbuf>>bitcnt)&bmask[sz]
         if bits<bhalf[sz]:bits-=bmask[sz]
         b16[n4]=bits*qt4_y[n4]
       k+=1
    while k<64:
     if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
     idx13=(bitbuf>>(bitcnt-13))&0x1FFF
     if k<=47:
      sk=yac_eob[idx13]
      if sk:bitcnt-=sk;break
     val=yac_f[idx13]
     if val>64:k+=((val>>5)&15)+1;bitcnt-=(val&31)
     elif val>0:
      if val<32:bitcnt-=val;break
      bitcnt-=(val-32);k+=16
     else:
      if val!=-999:neg=-val;bitcnt-=(neg&31);asym=neg>>5
      else:asym,wpos,bitbuf,bitcnt=_huff_slow(words,wpos,bitbuf,bitcnt,yac_s,13)
      if asym==0:break
      sz=asym&0xF
      if sz==0:
       if (asym>>4)==15:k+=16
       else:break
      else:
       k+=(asym>>4)+1
       if bitcnt<sz:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
       bitcnt-=sz
    d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15=b16;dc=d0*_c0;e2=d2*_c2;ev0=dc+e2;ev1=dc-e2;o0=d1*_h01+d3*_h03;o1=d1*_h11+d3*_h13;r00=ev0+o0;r01=ev1+o1;r02=ev1-o1;r03=ev0-o0;dc=d4*_c0;e2=d6*_c2;ev0=dc+e2;ev1=dc-e2;o0=d5*_h01+d7*_h03;o1=d5*_h11+d7*_h13;r10=ev0+o0;r11=ev1+o1;r12=ev1-o1;r13=ev0-o0;dc=d8*_c0;e2=d10*_c2;ev0=dc+e2;ev1=dc-e2;o0=d9*_h01+d11*_h03;o1=d9*_h11+d11*_h13;r20=ev0+o0;r21=ev1+o1;r22=ev1-o1;r23=ev0-o0;dc=d12*_c0;e2=d14*_c2;ev0=dc+e2;ev1=dc-e2;o0=d13*_h01+d15*_h03;o1=d13*_h11+d15*_h13;r30=ev0+o0;r31=ev1+o1;r32=ev1-o1;r33=ev0-o0;dc=r00*_c0;e2=r20*_c2;ev0=dc+e2;ev1=dc-e2;o0=r10*_h01+r30*_h03;o1=r10*_h11+r30*_h13;ybuf[by_off]=ct[int(ev0+o0+128.5)];ybuf[by_off+pw]=ct[int(ev1+o1+128.5)];ybuf[by_off+pw2]=ct[int(ev1-o1+128.5)];ybuf[by_off+pw3]=ct[int(ev0-o0+128.5)];dc=r01*_c0;e2=r21*_c2;ev0=dc+e2;ev1=dc-e2;o0=r11*_h01+r31*_h03;o1=r11*_h11+r31*_h13;b=by_off+1;ybuf[b]=ct[int(ev0+o0+128.5)];ybuf[b+pw]=ct[int(ev1+o1+128.5)];ybuf[b+pw2]=ct[int(ev1-o1+128.5)];ybuf[b+pw3]=ct[int(ev0-o0+128.5)];dc=r02*_c0;e2=r22*_c2;ev0=dc+e2;ev1=dc-e2;o0=r12*_h01+r32*_h03;o1=r12*_h11+r32*_h13;b=by_off+2;ybuf[b]=ct[int(ev0+o0+128.5)];ybuf[b+pw]=ct[int(ev1+o1+128.5)];ybuf[b+pw2]=ct[int(ev1-o1+128.5)];ybuf[b+pw3]=ct[int(ev0-o0+128.5)];dc=r03*_c0;e2=r23*_c2;ev0=dc+e2;ev1=dc-e2;o0=r13*_h01+r33*_h03;o1=r13*_h11+r33*_h13;b=by_off+3;ybuf[b]=ct[int(ev0+o0+128.5)];ybuf[b+pw]=ct[int(ev1+o1+128.5)];ybuf[b+pw2]=ct[int(ev1-o1+128.5)];ybuf[b+pw3]=ct[int(ev0-o0+128.5)]
   c_base=yb_row+mx*8
   for comp_i in (0,1):
    if comp_i==0:dc_f=cbdc_f;dc_s=cbdc_s;ac_f=cbac_f;ac_s=cbac_s;qt=qcb;pbuf=cbbuf;dc_prev=dc_cb;qt0=qcb0
    else:dc_f=crdc_f;dc_s=crdc_s;ac_f=crac_f;ac_s=crac_s;qt=qcr;pbuf=crbuf;dc_prev=dc_cr;qt0=qcr0
    if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
    dval=dc_f[(bitbuf>>(bitcnt-12))&0xFFF]
    if dval>=0:dc_prev+=(dval>>5)-4096;bitcnt-=(dval&31)
    else:
     if dval!=-999:neg=-dval;bitcnt-=(neg&31);sz=neg>>5
     else:sz,wpos,bitbuf,bitcnt=_huff_slow(words,wpos,bitbuf,bitcnt,dc_s,12)
     if sz:
      if bitcnt<sz:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
      bitcnt-=sz;bits=(bitbuf>>bitcnt)&bmask[sz]
      if bits<bhalf[sz]:bits-=bmask[sz]
      dc_prev+=bits
    if comp_i==0:dc_cb=dc_prev
    else:dc_cr=dc_prev
    dc_val=dc_prev*qt0
    if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
    val=ac_f[(bitbuf>>(bitcnt-13))&0x1FFF]
    if 0<val<32:bitcnt-=val;rb=fill8[ct[int(dc_val*0.125+128.5)]];b=c_base;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;continue
    blk[:]=z64;blk[0]=dc_val;nz_rows=1;nz_cols=1;k=1;first=True
    while k<64:
     if not first:
      if bitcnt<16:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
      val=ac_f[(bitbuf>>(bitcnt-13))&0x1FFF]
     first=False
     if val>64:
      k+=(val>>5)&15;bitcnt-=(val&31)
      if k<64:nat=zz8[k];blk[nat]=((val>>9)-4096)*qt[nat];nz_rows|=1<<(nat>>3);nz_cols|=1<<(nat&7)
      k+=1
     elif val>0:
      if val<32:bitcnt-=val;break
      bitcnt-=(val-32);k+=16
     else:
      if val!=-999:neg=-val;bitcnt-=(neg&31);asym=neg>>5
      else:asym,wpos,bitbuf,bitcnt=_huff_slow(words,wpos,bitbuf,bitcnt,ac_s,13)
      if asym==0:break
      rle=asym>>4;sz=asym&0xF
      if sz==0:
       if rle==15:k+=16
       else:break
      else:
       k+=rle
       if k<64:
        if bitcnt<sz:bitbuf=((bitbuf&0xFFFF)<<64)|words[wpos];wpos+=1;bitcnt+=64
        bitcnt-=sz;bits=(bitbuf>>bitcnt)&bmask[sz]
        if bits<bhalf[sz]:bits-=bmask[sz]
        nat=zz8[k];blk[nat]=bits*qt[nat];nz_rows|=1<<(nat>>3);nz_cols|=1<<(nat&7)
       k+=1
    if nz_rows==1:d0=blk[0]*_s0;d2=blk[2];d4=blk[4]*_s0;d6=blk[6];e0=d0+d4;e1=d0-d4;e2=d2*_s6-d6*_s2;e3=d2*_s2+d6*_s6;p0=e0+e3;p3=e0-e3;p1=e1+e2;p2=e1-e2;d1=blk[1];d3=blk[3];d5=blk[5];d7=blk[7];o0=d1*_s1+d3*_s3+d5*_s5+d7*_s7;o1=d1*_s3-d3*_s7-d5*_s1-d7*_s5;o2=d1*_s5-d3*_s1+d5*_s7+d7*_s3;o3=d1*_s7-d3*_s5+d5*_s3-d7*_s1;rb=bytes((ct[int((p0+o0)*_s0+128.5)],ct[int((p1+o1)*_s0+128.5)],ct[int((p2+o2)*_s0+128.5)],ct[int((p3+o3)*_s0+128.5)],ct[int((p3-o3)*_s0+128.5)],ct[int((p2-o2)*_s0+128.5)],ct[int((p1-o1)*_s0+128.5)],ct[int((p0-o0)*_s0+128.5)],));b=c_base;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb;b+=pw;pbuf[b:b+8]=rb
    elif (nz_rows&0xF0)==0:
     if (nz_cols&0xF0)==0:
      for r in range(4):
       i0=r<<3
       if (nz_rows>>r)&1:d0=blk[i0]*_s0;d2=blk[i0+2];e2=d2*_s6;e3=d2*_s2;p0=d0+e3;p3=d0-e3;p1=d0+e2;p2=d0-e2;d1=blk[i0+1];d3=blk[i0+3];o0=d1*_s1+d3*_s3;o1=d1*_s3-d3*_s7;o2=d1*_s5-d3*_s1;o3=d1*_s7-d3*_s5;tmp[i0]=p0+o0;tmp[i0+7]=p0-o0;tmp[i0+1]=p1+o1;tmp[i0+6]=p1-o1;tmp[i0+2]=p2+o2;tmp[i0+5]=p2-o2;tmp[i0+3]=p3+o3;tmp[i0+4]=p3-o3
       else:tmp[i0:i0+8]=z8
     else:
      for r in range(4):
       i0=r<<3
       if (nz_rows>>r)&1:d0=blk[i0]*_s0;d2=blk[i0+2];d4=blk[i0+4]*_s0;d6=blk[i0+6];e0=d0+d4;e1=d0-d4;e2=d2*_s6-d6*_s2;e3=d2*_s2+d6*_s6;p0=e0+e3;p3=e0-e3;p1=e1+e2;p2=e1-e2;d1=blk[i0+1];d3=blk[i0+3];d5=blk[i0+5];d7=blk[i0+7];o0=d1*_s1+d3*_s3+d5*_s5+d7*_s7;o1=d1*_s3-d3*_s7-d5*_s1-d7*_s5;o2=d1*_s5-d3*_s1+d5*_s7+d7*_s3;o3=d1*_s7-d3*_s5+d5*_s3-d7*_s1;tmp[i0]=p0+o0;tmp[i0+7]=p0-o0;tmp[i0+1]=p1+o1;tmp[i0+6]=p1-o1;tmp[i0+2]=p2+o2;tmp[i0+5]=p2-o2;tmp[i0+3]=p3+o3;tmp[i0+4]=p3-o3
       else:tmp[i0:i0+8]=z8
     for c in range(8):d0=tmp[c]*_s0;d2=tmp[c+16];e2=d2*_s6;e3=d2*_s2;p0=d0+e3;p3=d0-e3;p1=d0+e2;p2=d0-e2;d1=tmp[c+8];d3=tmp[c+24];o0=d1*_s1+d3*_s3;o1=d1*_s3-d3*_s7;o2=d1*_s5-d3*_s1;o3=d1*_s7-d3*_s5;idx=c_base+c;pbuf[idx]=ct[int(p0+o0+128.5)];idx+=pw;pbuf[idx]=ct[int(p1+o1+128.5)];idx+=pw;pbuf[idx]=ct[int(p2+o2+128.5)];idx+=pw;pbuf[idx]=ct[int(p3+o3+128.5)];idx+=pw;pbuf[idx]=ct[int(p3-o3+128.5)];idx+=pw;pbuf[idx]=ct[int(p2-o2+128.5)];idx+=pw;pbuf[idx]=ct[int(p1-o1+128.5)];idx+=pw;pbuf[idx]=ct[int(p0-o0+128.5)]
    else:
     for r in range(8):
      i0=r<<3
      if (nz_rows>>r)&1:d0=blk[i0]*_s0;d2=blk[i0+2];d4=blk[i0+4]*_s0;d6=blk[i0+6];e0=d0+d4;e1=d0-d4;e2=d2*_s6-d6*_s2;e3=d2*_s2+d6*_s6;p0=e0+e3;p3=e0-e3;p1=e1+e2;p2=e1-e2;d1=blk[i0+1];d3=blk[i0+3];d5=blk[i0+5];d7=blk[i0+7];o0=d1*_s1+d3*_s3+d5*_s5+d7*_s7;o1=d1*_s3-d3*_s7-d5*_s1-d7*_s5;o2=d1*_s5-d3*_s1+d5*_s7+d7*_s3;o3=d1*_s7-d3*_s5+d5*_s3-d7*_s1;tmp[i0]=p0+o0;tmp[i0+7]=p0-o0;tmp[i0+1]=p1+o1;tmp[i0+6]=p1-o1;tmp[i0+2]=p2+o2;tmp[i0+5]=p2-o2;tmp[i0+3]=p3+o3;tmp[i0+4]=p3-o3
      else:tmp[i0:i0+8]=z8
     for c in range(8):d0=tmp[c]*_s0;d2=tmp[c+16];d4=tmp[c+32]*_s0;d6=tmp[c+48];e0=d0+d4;e1=d0-d4;e2=d2*_s6-d6*_s2;e3=d2*_s2+d6*_s6;p0=e0+e3;p3=e0-e3;p1=e1+e2;p2=e1-e2;d1=tmp[c+8];d3=tmp[c+24];d5=tmp[c+40];d7=tmp[c+56];o0=d1*_s1+d3*_s3+d5*_s5+d7*_s7;o1=d1*_s3-d3*_s7-d5*_s1-d7*_s5;o2=d1*_s5-d3*_s1+d5*_s7+d7*_s3;o3=d1*_s7-d3*_s5+d5*_s3-d7*_s1;idx=c_base+c;pbuf[idx]=ct[int(p0+o0+128.5)];idx+=pw;pbuf[idx]=ct[int(p1+o1+128.5)];idx+=pw;pbuf[idx]=ct[int(p2+o2+128.5)];idx+=pw;pbuf[idx]=ct[int(p3+o3+128.5)];idx+=pw;pbuf[idx]=ct[int(p3-o3+128.5)];idx+=pw;pbuf[idx]=ct[int(p2-o2+128.5)];idx+=pw;pbuf[idx]=ct[int(p1-o1+128.5)];idx+=pw;pbuf[idx]=ct[int(p0-o0+128.5)]
 return (b"".join(ybuf[y*504:y*504+500]for y in range(500)),b"".join(cbbuf[y*504:y*504+500]for y in range(500)),b"".join(crbuf[y*504:y*504+500]for y in range(500)),)
_SCHEMA=(("rim_gate_W",(1,14)),("rim_gate_b",(1,)),("rim_gate_thr",(1,)),("rim0_W",(12,14)),("rim0_b",(12,)),("rim1_W",(8,12)),("rim1_b",(8,)),("rim2_W",(3,8)),("rim2_b",(3,)),("coarse0_W",(12,25)),("coarse0_b",(12,)),("coarse1_W",(8,12)),("coarse1_b",(8,)),("coarse2_W",(4,8)),("coarse2_b",(4,)),("W_cls",(3,4,4,16)),("acc0_W",(20,33)),("acc0_b",(20,)),("acc1_W",(10,20)),("acc1_b",(10,)),("acc2_W",(4,10)),("acc2_b",(4,)),("radii_500",(3,)),)
def load_weights(b85_str:str):
 blob=base64.b85decode(b85_str.encode("ascii"));n_tot,len_se,len_m8=struct.unpack_from("<HHH",blob,0);p_m8=166+len_se;p_lo=p_m8+len_m8;lo2_bytes=blob[p_lo:]
 def dec_huff(hdr,n_syms,bs,count):
  lens=[]
  for b in hdr:lens.append(b>>4);lens.append(b&0xF)
  pairs=sorted((l,s)for s,l in enumerate(lens[:n_syms])if l>0);lut=[-1]*32768;code=0;prev_l=0
  for l,s in pairs:code<<=(l-prev_l);sh=15-l;base=code<<sh;val=(s<<4)|l;lut[base:base+(1<<sh)]=[val]*(1<<sh);code+=1;prev_l=l
  out=[0]*count;bitbuf=int.from_bytes(bs+b"\x00\x00","big");tot_bits=(len(bs)+2)*8;bpos=0
  for i in range(count):v=lut[(bitbuf>>(tot_bits-15-bpos))&0x7FFF];out[i]=v>>4;bpos+=v&0xF
  return out
 se_stream=dec_huff(blob[6:38],64,blob[166:p_m8],n_tot);m8_stream=dec_huff(blob[38:166],256,blob[p_m8:p_lo],n_tot);u16_list=[0]*n_tot;idx=0
 for name,shp in _SCHEMA:
  sz=1
  for d in shp:sz*=d
  if name=="W_cls":
   for _ in range(sz//16):
    prev_se=0;prev_m8=0
    for _a in range(16):se=se_stream[idx];dm8=m8_stream[idx];m8=((prev_m8+dm8)&0xFF)if se==prev_se else dm8;lo2=(lo2_bytes[idx>>2]>>((3-(idx&3))*2))&3;u16_list[idx]=(se<<10)|(m8<<2)|lo2;prev_se=se;prev_m8=m8;idx+=1
  else:
   for _ in range(sz):se=se_stream[idx];m8=m8_stream[idx];lo2=(lo2_bytes[idx>>2]>>((3-(idx&3))*2))&3;u16_list[idx]=(se<<10)|(m8<<2)|lo2;idx+=1
 raw=struct.pack(f"<{n_tot}H",*u16_list);vals=struct.unpack(f"<{n_tot}e",raw);pos=0;w={}
 for name,shp in _SCHEMA:
  sz=1
  for d in shp:sz*=d
  sl=list(vals[pos:pos+sz]);pos+=sz
  if len(shp)==1:w[name]=sl
  elif len(shp)==2:r,c=shp;w[name]=[sl[i*c:(i+1)*c]for i in range(r)]
  else:
   out4=[];i4=0
   for k in range(3):
    c_list=[]
    for c in range(4):
     r_list=[]
     for r in range(4):r_list.append(sl[i4:i4+16]);i4+=16
     c_list.append(r_list)
    out4.append(c_list)
   w[name]=out4
 return w
def _RFFT16_RI(rows):
 D=len(rows);D2=D+D;F0=[0.0]*D;F8=[0.0]*D;fc1=[0.0]*D2;fc2=[0.0]*D2;fc3=[0.0]*D2;fc4=[0.0]*D2;fc5=[0.0]*D2;fc6=[0.0]*D2;fc7=[0.0]*D2;c2=_C2;c4=_C4;c6=_C6
 for p in range(D):q=p+D;x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15=rows[p];s0=x0+x8;d0=x0-x8;s1=x1+x9;d1=x1-x9;s2=x2+x10;d2=x2-x10;s3=x3+x11;d3=x3-x11;s4=x4+x12;d4=x4-x12;s5=x5+x13;d5=x5-x13;s6=x6+x14;d6=x6-x14;s7=x7+x15;d7=x7-x15;e0=s0+s4;f0=s0-s4;e1=s1+s5;f1=s1-s5;e2=s2+s6;f2=s2-s6;e3=s3+s7;f3=s3-s7;e02=e0+e2;e13=e1+e3;F0[p]=e02+e13;F8[p]=e02-e13;fc4[p]=e0-e2;fc4[q]=e3-e1;t2=(f1-f3)*c4;u2=(f1+f3)*c4;fc2[p]=f0+t2;fc2[q]=-f2-u2;fc6[p]=f0-t2;fc6[q]=f2-u2;p1=d1+d7;m1=d1-d7;p2=d2+d6;m2=d2-d6;p3=d3+d5;m3=d3-d5;d26=m2*c4;s26=p2*c4;a1=d0+d26;a2=d0-d26;b1=-d4-s26;b2=d4-s26;r13=m1*c2+m3*c6;r31=m1*c6-m3*c2;i13=p1*c6+p3*c2;i31=p1*c2-p3*c6;fc1[p]=a1+r13;fc1[q]=b1-i13;fc7[p]=a1-r13;fc7[q]=-b1-i13;fc3[p]=a2+r31;fc3[q]=b2-i31;fc5[p]=a2-r31;fc5[q]=-b2-i31
 return F0,fc1,fc2,fc3,fc4,fc5,fc6,fc7,F8
def _irfft16_unscaled(a0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7,a8):c2=_C2;c4=_C4;c6=_C6;s0=a0+a8;d0=a0-a8;s1=a1+a7;d1=a1-a7;s2=a2+a6;d2=a2-a6;s3=a3+a5;d3=a3-a5;s4=a4+a4;e0=s0+s4;f0=s0-s4;e02=e0+(s2+s2);e13=(s1+s3)*2.0;t2=(s1-s3)*(2.0*c4);E0=e02+e13;E8=e02-e13;E4=e0-(s2+s2);E2=f0+t2;E6=f0-t2;m1=d1+d1;m2=d2+d2;m3=d3+d3;d26=m2*c4;r13=m1*c2+m3*c6;r31=m1*c6-m3*c2;a1_=d0+d26;a2_=d0-d26;E1=a1_+r13;E7=a1_-r13;E3=a2_+r31;E5=a2_-r31;sb1=b1-b7;db1=b1+b7;sb2=b2-b6;db2=b2+b6;sb3=b3-b5;db3=b3+b5;d4=b4+b4;O4=(sb1-sb3)*2.0;u2=(sb1+sb3)*(2.0*c4);f2=sb2+sb2;O2=f2+u2;O6=u2-f2;p1=db1+db1;p2=db2+db2;p3=db3+db3;s26=p2*c4;i13=p1*c6+p3*c2;i31=p1*c2-p3*c6;b1_=s26+d4;b2_=s26-d4;O1=i13+b1_;O7=i13-b1_;O3=i31+b2_;O5=i31-b2_;return (E0,E1-O1,E2-O2,E3-O3,E4-O4,E5-O5,E6-O6,E7-O7,E8,E7+O7,E6+O6,E5+O5,E4+O4,E3+O3,E2+O2,E1+O1,)
_SWAR_01=int.from_bytes(b"\x01"*250000,"big");_SWAR_80=_SWAR_01<<7;_SWAR_7F=_SWAR_01*0x7F;_SWAR_F0=_SWAR_01*0xF0;_SWAR_0F=_SWAR_01*0x0F;_SWAR_117=_SWAR_01*117;_SWAR_70=_SWAR_01*0x70;_TRANS_FG=bytes(49 if b else 48 for b in range(256));_MASK500=(1<<500)-1;_MASK39=(1<<39)-1;_ONES_TBL=tuple(b"\x01"*i for i in range(48))
if hasattr(int,"bit_count"):
 def _popcnt(v:int)->int:return v.bit_count()
else:
 def _popcnt(v:int)->int:return bin(v).count("1")
_dot=getattr(math,"sumprod",None)
if _dot is None:
 def _dot(a,b):return sum(map(mul,a,b))
_AP16=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0);_AM16=(15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14)
def _fill_f3_masked(F0,vis,F3,F3_all):
 df0=0.0;df1=0.0;ap16=_AP16;am16=_AM16
 for r in range(4):
  vr=vis[r];Yr=F0[r];f3r=F3[r];vrp=vis[r+1]if r<3 else None;Yrp=F0[r+1]if r<3 else None;vrm=vis[r-1]if r>0 else None;Yrm=F0[r-1]if r>0 else None;fa3r=F3_all[r]if F3_all is not None else None;de=0.0
  for a in range(16):
   if not vr[a]:continue
   ap=ap16[a];am=am16[a];vp=vrp is not None and vrp[a];vm=vrm is not None and vrm[a];vap=vr[ap];vam=vr[am]
   if fa3r is not None and (vrp is None or vp)and (vrm is None or vm)and vap and vam:v3=fa3r[a];f3r[a]=v3;de+=v3*v3;continue
   if vp and vm:gr=0.5*(Yrp[a]-Yrm[a])
   elif vp:gr=Yrp[a]-Yr[a]
   elif vm:gr=Yr[a]-Yrm[a]
   else:gr=0.0
   if vap and vam:ga=0.5*(Yr[ap]-Yr[am])
   elif vap:ga=Yr[ap]-Yr[a]
   elif vam:ga=Yr[a]-Yr[am]
   else:ga=0.0
   v3=((gr if gr>=0.0 else -gr)+(ga if ga>=0.0 else -ga))*0.9-0.4;f3r[a]=v3;de+=v3*v3
  if r<2:df0+=de
  else:df1+=de
 return df0,df1
def _near_acc(ag,x:float,y:float,r2:float)->bool:
 gx=int(x+32.0)>>3;gy=int(y+32.0)>>3
 if gx<1:gx=1
 elif gx>70:gx=70
 if gy<1:gy=1
 elif gy>70:gy=70
 for oy in (-72,0,72):
  rb=gy*72+oy+gx
  for oi in (rb-1,rb,rb+1):
   c=ag[oi]
   if c is not None:
    for ax,ay in c:
     if (x-ax)*(x-ax)+(y-ay)*(y-ay)<r2:return True
 return False
def _add_acc(ag,x:float,y:float):
 gx=int(x+32.0)>>3;gy=int(y+32.0)>>3
 if gx<1:gx=1
 elif gx>70:gx=70
 if gy<1:gy=1
 elif gy>70:gy=70
 idx=gy*72+gx;c=ag[idx]
 if c is None:ag[idx]=[(x,y)]
 else:c.append((x,y))
class CoinDetector:
 def __init__(self,weights:dict):
  self.w=weights;self.rim_gate_w=weights["rim_gate_W"][0];self.rim_gate_b=weights["rim_gate_b"][0];self.rim_gate_thr=weights["rim_gate_thr"][0];self.rim_layers=[(weights["rim0_W"],weights["rim0_b"]),(weights["rim1_W"],weights["rim1_b"]),(weights["rim2_W"],weights["rim2_b"]),];self.coarse_layers=[(weights["coarse0_W"],weights["coarse0_b"]),(weights["coarse1_W"],weights["coarse1_b"]),(weights["coarse2_W"],weights["coarse2_b"]),];self.acc_layers=[(weights["acc0_W"],weights["acc0_b"]),(weights["acc1_W"],weights["acc1_b"]),(weights["acc2_W"],weights["acc2_b"]),];self.radii_500=weights["radii_500"];self.W_cls=weights["W_cls"];self.W_fft_0=[];self.W_fft_8=[];self.W_fft_ri=[];self.W_fft_nri=[];self.W2_fft_0=[];self.W2_fft_8=[];self.W2_fft_ri=[];self.W2_fft_nri=[];self.W_b0=[[]for _ in range(3)];self.W_full=[[]for _ in range(3)];self.W2_b0=[[]for _ in range(3)];self.W2_full=[[]for _ in range(3)];self.W2_hp=[[]for _ in range(3)];self.W2_hm=[[]for _ in range(3)];self.dw0_all=[[]for _ in range(3)];self.dw1_all=[[]for _ in range(3)];self.dw_all_all=[[]for _ in range(3)];self.dw_hp_all=[[]for _ in range(3)];self.dw_hm_all=[[]for _ in range(3)];self.dw_full_all=[[]for _ in range(3)]
  for cls in range(3):
   Wc=self.W_cls[cls];rows16=[Wc[c][r]for c in range(4)for r in range(4)];wf=_RFFT16_RI(rows16);self.W_fft_0.append(tuple(wf[0]));self.W_fft_8.append(tuple(wf[8]));self.W_fft_ri.append(tuple(tuple(wf[k])for k in range(1,8)));self.W_fft_nri.append(tuple(tuple(-v for v in wf[k][16:])+tuple(wf[k][:16])for k in range(1,8)));rows4=[[sum(Wc[c][r][a]**2 for c in range(4))for a in range(16)]for r in range(4)];w2f=_RFFT16_RI(rows4);self.W2_fft_0.append(tuple(w2f[0]));self.W2_fft_8.append(tuple(w2f[8]));self.W2_fft_ri.append(tuple(tuple(w2f[k])for k in range(1,8)));self.W2_fft_nri.append(tuple(tuple(-v for v in w2f[k][4:])+tuple(w2f[k][:4])for k in range(1,8)))
   for s in range(16):sp=(s+1)&15;sm=(s-1)&15;wb0=tuple(Wc[c][r][(a-s)&15]for c in range(4)for r in (0,1)for a in range(16));wb1=tuple(Wc[c][r][(a-s)&15]for c in range(4)for r in (2,3)for a in range(16));w2b0=tuple(sum(Wc[c][r][(a-s)&15]**2 for c in range(4))for r in (0,1)for a in range(16));w2b1=tuple(sum(Wc[c][r][(a-s)&15]**2 for c in range(4))for r in (2,3)for a in range(16));self.W_b0[cls].append(wb0);self.W_full[cls].append(wb0+wb1);self.W2_b0[cls].append(w2b0);self.W2_full[cls].append(w2b0+w2b1);w2hp=tuple(sum((0.5*(Wc[c][r][(a-s)&15]+Wc[c][r][(a-sp)&15]))**2 for c in range(4))for br in ((0,1),(2,3))for r in br for a in range(16));w2hm=tuple(sum((0.5*(Wc[c][r][(a-s)&15]+Wc[c][r][(a-sm)&15]))**2 for c in range(4))for br in ((0,1),(2,3))for r in br for a in range(16));self.W2_hp[cls].append(w2hp);self.W2_hm[cls].append(w2hm);dw0s=sum(w2b0)+1e-5;dw1s=sum(w2b1)+1e-5;self.dw0_all[cls].append(dw0s);self.dw1_all[cls].append(dw1s);self.dw_all_all[cls].append(dw0s+dw1s-1e-5);self.dw_full_all[cls].append(dw0s+dw1s-1e-5);self.dw_hp_all[cls].append(sum(w2hp)+1e-5);self.dw_hm_all[cls].append(sum(w2hm)+1e-5)
  (cw0,cb0),(cw1,cb1),(cw2,cb2)=self.coarse_layers;self._cw0=tuple(tuple(r)for r in cw0);self._cb0=tuple(cb0);self._cw1=tuple(tuple(r)for r in cw1);self._cb1=tuple(cb1);self._cdw=tuple(tuple(cw2[k][j]-cw2[0][j]for j in range(8))for k in (1,2,3));self._cdb=(cb2[1]-cb2[0],cb2[2]-cb2[0],cb2[3]-cb2[0]);(aw0,ab0),(aw1,ab1),(aw2,ab2)=self.acc_layers;self._aw0=tuple(tuple(r)for r in aw0);self._ab0=tuple(ab0);self._aw1=tuple(tuple(r)for r in aw1);self._ab1=tuple(ab1);self._adw=tuple(tuple(aw2[k][j]-aw2[0][j]for j in range(10))for k in (1,2,3));self._adb=(ab2[1]-ab2[0],ab2[2]-ab2[0],ab2[3]-ab2[0]);self._pt_cache={};self._eval_cache={}
 def _propose_centers(self,Y,Cb,Cr,fg,fg_rows=None,deadline=None):
  sumprod=_dot;(W0,b0),(W1,b1),(W2,b2)=self.rim_layers;w0_0,w0_1,w0_2,w0_3,w0_4,w0_5,w0_6,w0_7,w0_8,w0_9,w0_10,w0_11=W0;b0_0,b0_1,b0_2,b0_3,b0_4,b0_5,b0_6,b0_7,b0_8,b0_9,b0_10,b0_11=b0;w1_0,w1_1,w1_2,w1_3,w1_4,w1_5,w1_6,w1_7=W1;b1_0,b1_1,b1_2,b1_3,b1_4,b1_5,b1_6,b1_7=b1;w20,w21,w22=W2;b20,b21,b22=b2;acc=[0.0]*(536*536);nz=[];inv64=1.0/64.0;inv80=1.0/80.0;inv32=1.0/32.0;inv60=1.0/60.0;inv35=1.0/35.0;inv30=1.0/30.0;inv128=1.0/128.0;inv40=1.0/40.0;gw0,gw1,gw2,gw3,gw4,gw5,gw6,gw7,gw8,gw9,gw10,gw11,gw12,gw13=self.rim_gate_w;gb=self.rim_gate_b;gthr=self.rim_gate_thr;hypot=math.hypot;exp=math.exp
  for y in range(2,498,3):
   if deadline is not None and (y&31)==2 and time.perf_counter()>deadline:break
   if fg_rows is not None and not (fg_rows[y]|fg_rows[y-2]|fg_rows[y+2]):continue
   row=y*500;rm1=row-500;rp1=row+500;rm2=row-1000;rp2=row+1000
   for x in range(2,498,3):
    i=row+x
    if not (fg[i]or fg[i-2]or fg[i+2]or fg[rm2+x]or fg[rp2+x]):
     if Y[i]>235 and Y[i-1]>235 and Y[i+1]>235:continue
    ym2=Y[i-2];yp2=Y[i+2];ym2r=Y[rm2+x];yp2r=Y[rp2+x];gx1=Y[i+1]-Y[i-1];gy1=Y[rp1+x]-Y[rm1+x];gx2=yp2-ym2;gy2=yp2r-ym2r;cxb=Cb[i+2]-Cb[i-2];cyb=Cb[rp2+x]-Cb[rm2+x];cxr=Cr[i+2]-Cr[i-2];cyr=Cr[rp2+x]-Cr[rm2+x];graw=((gx1 if gx1>=0 else -gx1)+(gy1 if gy1>=0 else -gy1)+(gx2 if gx2>=0 else -gx2)+(gy2 if gy2>=0 else -gy2)+(((cxb if cxb>=0 else -cxb)+(cyb if cyb>=0 else -cyb)+(cxr if cxr>=0 else -cxr)+(cyr if cyr>=0 else -cyr))*2))
    if graw<46:continue
    yc=Y[i];f0=(yc-185.0)*inv60;f1=(Cb[i]-95.0)*inv35;f2=(Cr[i]-145.0)*inv30;f3=gx1*inv64;f4=gy1*inv64;f5=gx2*inv80;f6=gy2*inv80;f7=cxb*inv32;f8=cyb*inv32;f9=cxr*inv32;f10=cyr*inv32;f11=(yc*4-ym2-yp2-ym2r-yp2r)*inv128;mxnb=ym2 if ym2>yp2 else yp2
    if ym2r>mxnb:mxnb=ym2r
    if yp2r>mxnb:mxnb=yp2r
    f12=(mxnb-220.0)*inv40;f13=hypot(f5,f6)+hypot(f7,f8)+hypot(f9,f10);gl=(gw0*f0+gw1*f1+gw2*f2+gw3*f3+gw4*f4+gw5*f5+gw6*f6+gw7*f7+gw8*f8+gw9*f9+gw10*f10+gw11*f11+gw12*f12+gw13*f13+gb)
    if gl<gthr:continue
    fvec=(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13);h0=(v if (v:=sumprod(w0_0,fvec)+b0_0)>0.0 else 0.0,v if (v:=sumprod(w0_1,fvec)+b0_1)>0.0 else 0.0,v if (v:=sumprod(w0_2,fvec)+b0_2)>0.0 else 0.0,v if (v:=sumprod(w0_3,fvec)+b0_3)>0.0 else 0.0,v if (v:=sumprod(w0_4,fvec)+b0_4)>0.0 else 0.0,v if (v:=sumprod(w0_5,fvec)+b0_5)>0.0 else 0.0,v if (v:=sumprod(w0_6,fvec)+b0_6)>0.0 else 0.0,v if (v:=sumprod(w0_7,fvec)+b0_7)>0.0 else 0.0,v if (v:=sumprod(w0_8,fvec)+b0_8)>0.0 else 0.0,v if (v:=sumprod(w0_9,fvec)+b0_9)>0.0 else 0.0,v if (v:=sumprod(w0_10,fvec)+b0_10)>0.0 else 0.0,v if (v:=sumprod(w0_11,fvec)+b0_11)>0.0 else 0.0,);h1=(v if (v:=sumprod(w1_0,h0)+b1_0)>0.0 else 0.0,v if (v:=sumprod(w1_1,h0)+b1_1)>0.0 else 0.0,v if (v:=sumprod(w1_2,h0)+b1_2)>0.0 else 0.0,v if (v:=sumprod(w1_3,h0)+b1_3)>0.0 else 0.0,v if (v:=sumprod(w1_4,h0)+b1_4)>0.0 else 0.0,v if (v:=sumprod(w1_5,h0)+b1_5)>0.0 else 0.0,v if (v:=sumprod(w1_6,h0)+b1_6)>0.0 else 0.0,v if (v:=sumprod(w1_7,h0)+b1_7)>0.0 else 0.0,);lg=sumprod(w20,h1)+b20
    if lg<-1.2:continue
    vx=sumprod(w21,h1)+b21;vy=sumprod(w22,h1)+b22;nv=hypot(vx,vy)
    if nv<0.25:continue
    pr=1.0/(1.0+exp(-lg));ux=vx/nv;uy=vy/nv
    for rad in (18.3,18.6,18.9):
     cx=int(x+rad*ux+18.5);cy=int(y+rad*uy+18.5)
     if 0<=cx<536 and 0<=cy<536:
      p=cy*536+cx;old_a=acc[p];nv_a=old_a+pr;acc[p]=nv_a
      if old_a<0.70 and nv_a>=0.70:nz.append(p)
  cands=[];sm={}
  for idx in nz:
   y=idx//536;x=idx-y*536
   if 1<=x<535 and 1<=y<535:
    r0=(y-1)*536+x;r1=r0+536;r2=r1+536;s=acc[r0-1]+acc[r0]+acc[r0+1]+acc[r1-1]+acc[r1]+acc[r1+1]+acc[r2-1]+acc[r2]+acc[r2+1];thr=1.1 if (x<34 or x>502 or y<34 or y>502)else 2.8
    if s>=thr:sm[(x,y)]=s
  for (x,y),s in sorted(sm.items(),key=lambda kv:-kv[1]):
   is_pk=True
   for dy in (-2,-1,0,1,2):
    for dx in (-2,-1,0,1,2):
     if (dx or dy)and sm.get((x+dx,y+dy),0.0)>s:is_pk=False;break
    if not is_pk:break
   if is_pk:
    sw=sxw=syw=0.0
    for dy in (-2,-1,0,1,2):
     yi=y+dy
     if 0<=yi<536:
      ry=yi*536
      for dx in (-2,-1,0,1,2):
       xi=x+dx
       if 0<=xi<536:
        w=acc[ry+xi]
        if w>0.0:sw+=w;sxw+=w*xi;syw+=w*yi
    cands.append(((sxw/sw-18.0)if sw>0.0 else float(x-18),(syw/sw-18.0)if sw>0.0 else float(y-18),s))
  return cands
 def _coarse_score(self,Y,Cb,Cr,fg,claimed,cx:float,cy:float):
  feats=[0.0]*25;tot_vis=tot_fg=tot_oob=0;sqrt=math.sqrt;y2_tbl=_Y2_TBL
  if 20.5<=cx<=478.5 and 20.5<=cy<=478.5:
   ix0=int(cx);iy0=int(cy)
   if cx==ix0 and cy==iy0:
    cpos=iy0*500+ix0;off=0
    for ioffs in _COARSE_IOFF:
     nv=r_fg=sy=sy2=scb=scr=0
     for dpos in ioffs:
      pos=cpos+dpos
      if not claimed[pos]:nv+=1;yv=Y[pos];sy+=yv;sy2+=y2_tbl[yv];scb+=Cb[pos];scr+=Cr[pos];r_fg+=fg[pos]
     if nv:tot_vis+=nv;tot_fg+=r_fg;inv=1.0/nv;my=sy*inv;feats[off]=(my-185.0)*(1.0/45.0);feats[off+1]=(scb*inv-92.0)*(1.0/25.0);feats[off+2]=(scr*inv-148.0)*(1.0/25.0);v=sy2*inv-my*my;feats[off+3]=(sqrt(v)if v>0.0 else 0.0)*(1.0/35.0);feats[off+4]=r_fg*inv
     off+=5
    if tot_fg<4:return 0.0,tot_fg
    soy=obg_cnt=0
    for dpos in _OUTER_IOFF:
     pos=cpos+dpos;soy+=Y[pos]
     if not fg[pos]or claimed[pos]:obg_cnt+=1
   else:
    off=0;cx5=cx+0.5;cy5=cy+0.5
    for ring_pts in _COARSE_OFF:
     nv=r_fg=sy=sy2=scb=scr=0
     for dx,dy in ring_pts:
      pos=int(cy5+dy)*500+int(cx5+dx)
      if not claimed[pos]:nv+=1;yv=Y[pos];sy+=yv;sy2+=y2_tbl[yv];scb+=Cb[pos];scr+=Cr[pos];r_fg+=fg[pos]
     if nv:tot_vis+=nv;tot_fg+=r_fg;inv=1.0/nv;my=sy*inv;feats[off]=(my-185.0)*(1.0/45.0);feats[off+1]=(scb*inv-92.0)*(1.0/25.0);feats[off+2]=(scr*inv-148.0)*(1.0/25.0);v=sy2*inv-my*my;feats[off+3]=(sqrt(v)if v>0.0 else 0.0)*(1.0/35.0);feats[off+4]=r_fg*inv
     off+=5
    if tot_fg<4:return 0.0,tot_fg
    soy=obg_cnt=0
    for dx,dy in _OUTER_OFF:
     pos=int(cy5+dy)*500+int(cx5+dx);soy+=Y[pos]
     if not fg[pos]or claimed[pos]:obg_cnt+=1
   omean_y=((soy*(1.0/12.0))-210.0)*(1.0/50.0);obg=obg_cnt*(1.0/12.0)
  else:
   off=0;cx5=cx+0.5;cy5=cy+0.5
   for ring_pts in _COARSE_OFF:
    nv=r_fg=sy=sy2=scb=scr=0
    for dx,dy in ring_pts:
     ix=int(cx5+dx);iy=int(cy5+dy)
     if 0<=ix<500 and 0<=iy<500:
      pos=iy*500+ix
      if not claimed[pos]:nv+=1;yv=Y[pos];sy+=yv;sy2+=y2_tbl[yv];scb+=Cb[pos];scr+=Cr[pos];r_fg+=fg[pos]
     else:tot_oob+=1
    if nv:tot_vis+=nv;tot_fg+=r_fg;inv=1.0/nv;my=sy*inv;feats[off]=(my-185.0)*(1.0/45.0);feats[off+1]=(scb*inv-92.0)*(1.0/25.0);feats[off+2]=(scr*inv-148.0)*(1.0/25.0);v=sy2*inv-my*my;feats[off+3]=(sqrt(v)if v>0.0 else 0.0)*(1.0/35.0);feats[off+4]=r_fg*inv
    off+=5
   if tot_fg<4:return 0.0,tot_fg
   soy=obg_cnt=oin=0
   for dx,dy in _OUTER_OFF:
    ix=int(cx5+dx);iy=int(cy5+dy)
    if 0<=ix<500 and 0<=iy<500:
     oin+=1;pos=iy*500+ix;soy+=Y[pos]
     if not fg[pos]or claimed[pos]:obg_cnt+=1
   omean_y=((soy/oin)-210.0)*(1.0/50.0)if oin>0 else 0.0;obg=(obg_cnt/oin)if oin>0 else 1.0
  inv34=1.0/34.0;feats[20]=tot_vis*inv34;feats[21]=tot_fg*inv34;feats[22]=tot_oob*inv34;feats[23]=omean_y;feats[24]=obg;dot=_dot;exp=math.exp;c0_0,c0_1,c0_2,c0_3,c0_4,c0_5,c0_6,c0_7,c0_8,c0_9,c0_10,c0_11=self._cw0;cb0_0,cb0_1,cb0_2,cb0_3,cb0_4,cb0_5,cb0_6,cb0_7,cb0_8,cb0_9,cb0_10,cb0_11=self._cb0;c1_0,c1_1,c1_2,c1_3,c1_4,c1_5,c1_6,c1_7=self._cw1;cb1_0,cb1_1,cb1_2,cb1_3,cb1_4,cb1_5,cb1_6,cb1_7=self._cb1;cd1,cd2,cd3=self._cdw;cdb1,cdb2,cdb3=self._cdb;h0=(v if (v:=dot(c0_0,feats)+cb0_0)>0.0 else 0.0,v if (v:=dot(c0_1,feats)+cb0_1)>0.0 else 0.0,v if (v:=dot(c0_2,feats)+cb0_2)>0.0 else 0.0,v if (v:=dot(c0_3,feats)+cb0_3)>0.0 else 0.0,v if (v:=dot(c0_4,feats)+cb0_4)>0.0 else 0.0,v if (v:=dot(c0_5,feats)+cb0_5)>0.0 else 0.0,v if (v:=dot(c0_6,feats)+cb0_6)>0.0 else 0.0,v if (v:=dot(c0_7,feats)+cb0_7)>0.0 else 0.0,v if (v:=dot(c0_8,feats)+cb0_8)>0.0 else 0.0,v if (v:=dot(c0_9,feats)+cb0_9)>0.0 else 0.0,v if (v:=dot(c0_10,feats)+cb0_10)>0.0 else 0.0,v if (v:=dot(c0_11,feats)+cb0_11)>0.0 else 0.0,);h1=(v if (v:=dot(c1_0,h0)+cb1_0)>0.0 else 0.0,v if (v:=dot(c1_1,h0)+cb1_1)>0.0 else 0.0,v if (v:=dot(c1_2,h0)+cb1_2)>0.0 else 0.0,v if (v:=dot(c1_3,h0)+cb1_3)>0.0 else 0.0,v if (v:=dot(c1_4,h0)+cb1_4)>0.0 else 0.0,v if (v:=dot(c1_5,h0)+cb1_5)>0.0 else 0.0,v if (v:=dot(c1_6,h0)+cb1_6)>0.0 else 0.0,v if (v:=dot(c1_7,h0)+cb1_7)>0.0 else 0.0,);d1=dot(cd1,h1)+cdb1;d2=dot(cd2,h1)+cdb2;d3=dot(cd3,h1)+cdb3;dm=d1 if d1>d2 else d2
  if d3>dm:dm=d3
  if dm<-4.5:return 0.0,tot_fg
  if dm>14.0:return 1.0,tot_fg
  s=exp(d1)+exp(d2)+exp(d3);return s/(1.0+s),tot_fg
 def _extract_features(self,Y,Cb,Cr,fg,claimed,cx:float,cy:float,need_full:bool=True):
  ckey=(cx,cy);ent=self._pt_cache.get(ckey)
  if ent is not None:
   pos_rows,fg_rows_p,opos_fg,oin,n_inb,F0c,F1c,F2c,F3_all,last_nv,last_obg,last_st=ent;nv_now=0
   for pr in pos_rows:
    for p in pr:
     if p>=0 and not claimed[p]:nv_now+=1
   if nv_now<6:ent[9]=nv_now;ent[11]=None;return None
   obg_uncl=0
   for p in opos_fg:
    if not claimed[p]:obg_uncl+=1
   obg_cnt=oin-obg_uncl
   if nv_now==last_nv:
    if last_st is None:return None
    if obg_cnt==last_obg:
     if need_full and last_st[3]is None:F0,F1,F2,F3=last_st[0];F_full=[*last_st[2],*F0[2],*F0[3],*F1[2],*F1[3],*F2[2],*F2[3],*F3[2],*F3[3]];last_st=(last_st[0],last_st[1],last_st[2],F_full,*last_st[4:]);ent[11]=last_st
     return last_st
    obg=(obg_cnt/oin)if oin>0 else 1.0;st12=last_st[6][:];st12[11]=obg;F_full=last_st[3]
    if need_full and F_full is None:F0,F1,F2,F3=last_st[0];F_full=[*last_st[2],*F0[2],*F0[3],*F1[2],*F1[3],*F2[2],*F2[3],*F3[2],*F3[3]]
    new_st=(last_st[0],last_st[1],last_st[2],F_full,last_st[4],last_st[5],st12,last_st[7],last_st[8],last_st[9],last_st[10],last_st[11]);ent[10]=obg_cnt;ent[11]=new_st;return new_st
   vis=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16];F0=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16];F1=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16];F2=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16];F3=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16];s0=s0_2=s1=s1_2=s2=s2_2=0.0;df0=1e-5;df1=1e-5;n_vis=n_fg_vis=n_in_vis=n_in_fg=0
   for r in range(4):
    if r==2:n_in_vis=n_vis;n_in_fg=n_fg_vis
    pr=pos_rows[r];fgr=fg_rows_p[r];vr=vis[r];f0r=F0[r];f1r=F1[r];f2r=F2[r];c0r=F0c[r];c1r=F1c[r];c2r=F2c[r];de=0.0
    for a in range(16):
     p=pr[a]
     if p>=0 and not claimed[p]:
      vr[a]=1.0;n_vis+=1;v0=c0r[a];v1=c1r[a];v2=c2r[a];f0r[a]=v0;f1r[a]=v1;f2r[a]=v2;q0=v0*v0;q1=v1*v1;q2=v2*v2;s0+=v0;s0_2+=q0;s1+=v1;s1_2+=q1;s2+=v2;s2_2+=q2;de+=q0+q1+q2
      if fgr[a]:n_fg_vis+=1
    if r<2:df0+=de
    else:df1+=de
   if n_fg_vis<4:ent[9]=n_vis;ent[10]=obg_cnt;ent[11]=None;return None
   e0,e1=_fill_f3_masked(F0c,vis,F3,F3_all);df0+=e0;df1+=e1;n_rm_vis=n_vis-n_in_vis;n_rm_fg=n_fg_vis-n_in_fg;obg=(obg_cnt/oin)if oin>0 else 1.0;inv_vis=1.0/n_vis;m0=s0*inv_vis;m1=s1*inv_vis;m2=s2*inv_vis;sqrt=math.sqrt;v0v=s0_2*inv_vis-m0*m0;v1v=s1_2*inv_vis-m1*m1;v2v=s2_2*inv_vis-m2*m2;inv64=1.0/64.0;stats12=[n_vis*inv64,n_inb*inv64,n_fg_vis*inv_vis,(n_in_fg/n_in_vis)if n_in_vis>0 else 0.0,(n_rm_fg/n_rm_vis)if n_rm_vis>0 else 0.0,m0,m1,m2,(sqrt(v0v)if v0v>0.0 else 0.0)*1.125,(sqrt(v1v)if v1v>0.0 else 0.0)*(25.0/15.0),(sqrt(v2v)if v2v>0.0 else 0.0)*(25.0/15.0),obg,];F=(F0,F1,F2,F3);F_b0=[*F0[0],*F0[1],*F1[0],*F1[1],*F2[0],*F2[1],*F3[0],*F3[1]];F_full=[*F_b0,*F0[2],*F0[3],*F1[2],*F1[3],*F2[2],*F2[3],*F3[2],*F3[3]]if need_full else None;V_b0=[*vis[0],*vis[1]];V_full=[*V_b0,*vis[2],*vis[3]];denF=df0+df1-1e-5;new_st=(F,vis,F_b0,F_full,V_b0,V_full,stats12,df0,df1,denF,n_fg_vis,False);ent[9]=n_vis;ent[10]=obg_cnt;ent[11]=new_st;return new_st
  save_c=not need_full;F0=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16];F1=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16];F2=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16];F3=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16]
  if save_c:pos_rows=[[-1]*16,[-1]*16,[-1]*16,[-1]*16];fg_rows_p=[[0]*16,[0]*16,[0]*16,[0]*16]
  s0=s0_2=s1=s1_2=s2=s2_2=0.0;df0=1e-5;df1=1e-5;n_vis=n_inb=n_fg_vis=n_in_vis=n_in_fg=0;polar_off=_POLAR_OFF;inv45=1.0/45.0;inv25=1.0/25.0
  if 18.0<=cx<=480.0 and 18.0<=cy<=480.0:
   vis=[[1.0]*16,[1.0]*16,[1.0]*16,[1.0]*16];n_inb=64
   for r in range(4):
    if r==2:n_in_vis=n_vis;n_in_fg=n_fg_vis
    row_off=polar_off[r];vr=vis[r];f0r=F0[r];f1r=F1[r];f2r=F2[r];de=0.0
    if save_c:
     pr=pos_rows[r];fgr=fg_rows_p[r]
     for a in range(16):
      dx,dy=row_off[a];px=cx+dx;py=cy+dy;x0=int(px);y0=int(py);fx=px-x0;fy=py-y0;i00=y0*500+x0;pos=i00+(1 if fx>=0.5 else 0)+(500 if fy>=0.5 else 0)
      if not claimed[pos]:
       pr[a]=pos;i10=i00+1;i01=i00+500;i11=i01+1;omx=1.0-fx;omy=1.0-fy;w00=omx*omy;w10=fx*omy;w01=omx*fy;w11=fx*fy;v0=(Y[i00]*w00+Y[i10]*w10+Y[i01]*w01+Y[i11]*w11-185.0)*inv45;v1=(Cb[i00]*w00+Cb[i10]*w10+Cb[i01]*w01+Cb[i11]*w11-92.0)*inv25;v2=(Cr[i00]*w00+Cr[i10]*w10+Cr[i01]*w01+Cr[i11]*w11-148.0)*inv25;f0r[a]=v0;f1r[a]=v1;f2r[a]=v2;n_vis+=1;q0=v0*v0;q1=v1*v1;q2=v2*v2;s0+=v0;s0_2+=q0;s1+=v1;s1_2+=q1;s2+=v2;s2_2+=q2;de+=q0+q1+q2
       if fg[pos]:n_fg_vis+=1;fgr[a]=1
      else:vr[a]=0.0
    else:
     for a in range(16):
      dx,dy=row_off[a];px=cx+dx;py=cy+dy;x0=int(px);y0=int(py);fx=px-x0;fy=py-y0;i00=y0*500+x0;pos=i00+(1 if fx>=0.5 else 0)+(500 if fy>=0.5 else 0)
      if not claimed[pos]:
       i10=i00+1;i01=i00+500;i11=i01+1;omx=1.0-fx;omy=1.0-fy;w00=omx*omy;w10=fx*omy;w01=omx*fy;w11=fx*fy;v0=(Y[i00]*w00+Y[i10]*w10+Y[i01]*w01+Y[i11]*w11-185.0)*inv45;v1=(Cb[i00]*w00+Cb[i10]*w10+Cb[i01]*w01+Cb[i11]*w11-92.0)*inv25;v2=(Cr[i00]*w00+Cr[i10]*w10+Cr[i01]*w01+Cr[i11]*w11-148.0)*inv25;f0r[a]=v0;f1r[a]=v1;f2r[a]=v2;n_vis+=1;q0=v0*v0;q1=v1*v1;q2=v2*v2;s0+=v0;s0_2+=q0;s1+=v1;s1_2+=q1;s2+=v2;s2_2+=q2;de+=q0+q1+q2
       if fg[pos]:n_fg_vis+=1
      else:vr[a]=0.0
    if r<2:df0+=de
    else:df1+=de
  else:
   vis=[[0.0]*16,[0.0]*16,[0.0]*16,[0.0]*16]
   for r in range(4):
    if r==2:n_in_vis=n_vis;n_in_fg=n_fg_vis
    row_off=polar_off[r];vr=vis[r];pr=pos_rows[r]if save_c else None;fgr=fg_rows_p[r]if save_c else None;f0r=F0[r];f1r=F1[r];f2r=F2[r];de=0.0
    for a in range(16):
     dx,dy=row_off[a];px=cx+dx;py=cy+dy
     if 0.0<=px<=499.0 and 0.0<=py<=499.0:
      n_inb+=1;pos=int(py+0.5)*500+int(px+0.5)
      if not claimed[pos]:
       if save_c:pr[a]=pos
       x0=int(px);y0=int(py);fx=px-x0;fy=py-y0
       if x0>498:x0=498;fx=px-498.0
       if y0>498:y0=498;fy=py-498.0
       omx=1.0-fx;omy=1.0-fy;w00=omx*omy;w10=fx*omy;w01=omx*fy;w11=fx*fy;i00=y0*500+x0;i10=i00+1;i01=i00+500;i11=i01+1;v0=(Y[i00]*w00+Y[i10]*w10+Y[i01]*w01+Y[i11]*w11-185.0)*inv45;v1=(Cb[i00]*w00+Cb[i10]*w10+Cb[i01]*w01+Cb[i11]*w11-92.0)*inv25;v2=(Cr[i00]*w00+Cr[i10]*w10+Cr[i01]*w01+Cr[i11]*w11-148.0)*inv25;f0r[a]=v0;f1r[a]=v1;f2r[a]=v2;vr[a]=1.0;n_vis+=1;q0=v0*v0;q1=v1*v1;q2=v2*v2;s0+=v0;s0_2+=q0;s1+=v1;s1_2+=q1;s2+=v2;s2_2+=q2;de+=q0+q1+q2
       if fg[pos]:
        n_fg_vis+=1
        if save_c:fgr[a]=1
    if r<2:df0+=de
    else:df1+=de
  if n_vis<6 or n_fg_vis<4:
   if save_c:self._pt_cache[ckey]=[pos_rows,fg_rows_p,(),0,n_inb,F0,F1,F2,None,n_vis,0,None]
   return None
  n_rm_vis=n_vis-n_in_vis;n_rm_fg=n_fg_vis-n_in_fg;opos_fg=[]if save_c else None;obg_cnt=0;cx5=cx+0.5;cy5=cy+0.5
  if 20.5<=cx<=478.5 and 20.5<=cy<=478.5:
   oin=12
   for dx,dy in _OUTER_OFF:
    pos=int(cy5+dy)*500+int(cx5+dx)
    if not fg[pos]or claimed[pos]:obg_cnt+=1
    elif save_c:opos_fg.append(pos)
   obg=obg_cnt*(1.0/12.0)
  else:
   oin=0
   for dx,dy in _OUTER_OFF:
    ix=int(cx5+dx);iy=int(cy5+dy)
    if 0<=ix<500 and 0<=iy<500:
     oin+=1;pos=iy*500+ix
     if not fg[pos]or claimed[pos]:obg_cnt+=1
     elif save_c:opos_fg.append(pos)
   obg=(obg_cnt/oin)if oin>0 else 1.0
  inv_vis=1.0/n_vis;m0=s0*inv_vis;m1=s1*inv_vis;m2=s2*inv_vis;sqrt=math.sqrt;v0v=s0_2*inv_vis-m0*m0;v1v=s1_2*inv_vis-m1*m1;v2v=s2_2*inv_vis-m2*m2;inv64=1.0/64.0;stats12=[n_vis*inv64,n_inb*inv64,n_fg_vis*inv_vis,(n_in_fg/n_in_vis)if n_in_vis>0 else 0.0,(n_rm_fg/n_rm_vis)if n_rm_vis>0 else 0.0,m0,m1,m2,(sqrt(v0v)if v0v>0.0 else 0.0)*1.125,(sqrt(v1v)if v1v>0.0 else 0.0)*(25.0/15.0),(sqrt(v2v)if v2v>0.0 else 0.0)*(25.0/15.0),obg,];all_vis=(n_vis==64)
  if all_vis:
   for r in range(4):
    Yr=F0[r];f3r=F3[r];de=0.0
    if r==0:Yrp=F0[1];Yrm=Yr;rsc=0.9
    elif r==3:Yrp=Yr;Yrm=F0[2];rsc=0.9
    else:Yrp=F0[r+1];Yrm=F0[r-1];rsc=0.45
    for a in range(16):gr=Yrp[a]-Yrm[a];ga=Yr[(a+1)&15]-Yr[(a-1)&15];v3=(gr if gr>=0.0 else -gr)*rsc+(ga if ga>=0.0 else -ga)*0.45-0.4;f3r[a]=v3;de+=v3*v3
    if r<2:df0+=de
    else:df1+=de
   V_b0=V_full=None;F3_all_save=F3
  else:e0,e1=_fill_f3_masked(F0,vis,F3,None);df0+=e0;df1+=e1;V_b0=[*vis[0],*vis[1]];V_full=[*V_b0,*vis[2],*vis[3]];F3_all_save=None
  F=(F0,F1,F2,F3);F_b0=[*F0[0],*F0[1],*F1[0],*F1[1],*F2[0],*F2[1],*F3[0],*F3[1]];F_full=[*F_b0,*F0[2],*F0[3],*F1[2],*F1[3],*F2[2],*F2[3],*F3[2],*F3[3]]if need_full else None;denF=df0+df1-1e-5;res_st=(F,vis,F_b0,F_full,V_b0,V_full,stats12,df0,df1,denF,n_fg_vis,all_vis)
  if save_c:self._pt_cache[ckey]=[pos_rows,fg_rows_p,tuple(opos_fg),oin,n_inb,F0,F1,F2,F3_all_save,n_vis,obg_cnt,res_st]
  return res_st
 def _finish_accept(self,F_b0,V_b0,V_full,stats12,df0,df1,denF,n_fg_vis,s_best,sim_best,sim_mean,nm_triples,dw_bests,all_vis:bool):
  acc_feat=[0.0]*33;acc_feat[21:33]=stats12;sqrt=math.sqrt;dot=_dot;log=math.log;sb0,sb1,sb2=sim_best;others=(sb1 if sb1>sb2 else sb2,sb0 if sb0>sb2 else sb2,sb0 if sb0>sb1 else sb1)
  for k in range(3):
   sk=s_best[k];nm_k,nm_kp,nm_km=nm_triples[k];nm0=dot(F_b0,self.W_b0[k][sk]);nm1=nm_k-nm0
   if all_vis:dw0=self.dw0_all[k][sk];dw1=self.dw1_all[k][sk];dw_all=self.dw_all_all[k][sk];dw_hp=self.dw_hp_all[k][sk];dw_hm=self.dw_hm_all[k][sk]
   else:
    dw0=dot(V_b0,self.W2_b0[k][sk])+1e-5;dw_all=dw_bests[k];dw1=dw_all-dw0+1e-5
    if dw1<1e-5:dw1=1e-5
    dw_hp=dot(V_full,self.W2_hp[k][sk])+1e-5;dw_hm=dot(V_full,self.W2_hm[k][sk])+1e-5
   c_sub_p=(0.5*(nm_k+nm_kp))/sqrt(denF*dw_hp);c_sub_m=(0.5*(nm_k+nm_km))/sqrt(denF*dw_hm);sbk=sim_best[k];sim_sub=sbk if sbk>c_sub_p else c_sub_p
   if c_sub_m>sim_sub:sim_sub=c_sub_m
   le=0.5*log(denF/dw_all)
   if le>3.0:le=3.0
   elif le<-3.0:le=-3.0
   off=k*7;acc_feat[off+0]=sbk;acc_feat[off+1]=sim_sub;acc_feat[off+2]=sbk-sim_mean[k];acc_feat[off+3]=sbk-others[k];acc_feat[off+4]=nm0/sqrt(df0*dw0);acc_feat[off+5]=nm1/sqrt(df1*dw1);acc_feat[off+6]=le
  a0_0,a0_1,a0_2,a0_3,a0_4,a0_5,a0_6,a0_7,a0_8,a0_9,a0_10,a0_11,a0_12,a0_13,a0_14,a0_15,a0_16,a0_17,a0_18,a0_19=self._aw0;ab0_0,ab0_1,ab0_2,ab0_3,ab0_4,ab0_5,ab0_6,ab0_7,ab0_8,ab0_9,ab0_10,ab0_11,ab0_12,ab0_13,ab0_14,ab0_15,ab0_16,ab0_17,ab0_18,ab0_19=self._ab0;a1_0,a1_1,a1_2,a1_3,a1_4,a1_5,a1_6,a1_7,a1_8,a1_9=self._aw1;ab1_0,ab1_1,ab1_2,ab1_3,ab1_4,ab1_5,ab1_6,ab1_7,ab1_8,ab1_9=self._ab1;ad1,ad2,ad3=self._adw;adb1,adb2,adb3=self._adb;h0=(v if (v:=dot(a0_0,acc_feat)+ab0_0)>0.0 else 0.0,v if (v:=dot(a0_1,acc_feat)+ab0_1)>0.0 else 0.0,v if (v:=dot(a0_2,acc_feat)+ab0_2)>0.0 else 0.0,v if (v:=dot(a0_3,acc_feat)+ab0_3)>0.0 else 0.0,v if (v:=dot(a0_4,acc_feat)+ab0_4)>0.0 else 0.0,v if (v:=dot(a0_5,acc_feat)+ab0_5)>0.0 else 0.0,v if (v:=dot(a0_6,acc_feat)+ab0_6)>0.0 else 0.0,v if (v:=dot(a0_7,acc_feat)+ab0_7)>0.0 else 0.0,v if (v:=dot(a0_8,acc_feat)+ab0_8)>0.0 else 0.0,v if (v:=dot(a0_9,acc_feat)+ab0_9)>0.0 else 0.0,v if (v:=dot(a0_10,acc_feat)+ab0_10)>0.0 else 0.0,v if (v:=dot(a0_11,acc_feat)+ab0_11)>0.0 else 0.0,v if (v:=dot(a0_12,acc_feat)+ab0_12)>0.0 else 0.0,v if (v:=dot(a0_13,acc_feat)+ab0_13)>0.0 else 0.0,v if (v:=dot(a0_14,acc_feat)+ab0_14)>0.0 else 0.0,v if (v:=dot(a0_15,acc_feat)+ab0_15)>0.0 else 0.0,v if (v:=dot(a0_16,acc_feat)+ab0_16)>0.0 else 0.0,v if (v:=dot(a0_17,acc_feat)+ab0_17)>0.0 else 0.0,v if (v:=dot(a0_18,acc_feat)+ab0_18)>0.0 else 0.0,v if (v:=dot(a0_19,acc_feat)+ab0_19)>0.0 else 0.0,);h1=(v if (v:=dot(a1_0,h0)+ab1_0)>0.0 else 0.0,v if (v:=dot(a1_1,h0)+ab1_1)>0.0 else 0.0,v if (v:=dot(a1_2,h0)+ab1_2)>0.0 else 0.0,v if (v:=dot(a1_3,h0)+ab1_3)>0.0 else 0.0,v if (v:=dot(a1_4,h0)+ab1_4)>0.0 else 0.0,v if (v:=dot(a1_5,h0)+ab1_5)>0.0 else 0.0,v if (v:=dot(a1_6,h0)+ab1_6)>0.0 else 0.0,v if (v:=dot(a1_7,h0)+ab1_7)>0.0 else 0.0,v if (v:=dot(a1_8,h0)+ab1_8)>0.0 else 0.0,v if (v:=dot(a1_9,h0)+ab1_9)>0.0 else 0.0,);d1=dot(ad1,h1)+adb1;d2=dot(ad2,h1)+adb2;d3=dot(ad3,h1)+adb3;mx=0.0
  if d1>mx:mx=d1
  if d2>mx:mx=d2
  if d3>mx:mx=d3
  exp=math.exp;e0=exp(-mx);e1=exp(d1-mx);e2=exp(d2-mx);e3=exp(d3-mx);invs=1.0/(e0+e1+e2+e3);probs=(e0*invs,e1*invs,e2*invs,e3*invs);p_coin=1.0-probs[0];p1=probs[1];p2=probs[2];p3=probs[3];cls=0 if (p1>=p2 and p1>=p3)else (1 if p2>=p3 else 2);d_c=d1 if cls==0 else (d2 if cls==1 else d3);snap_sc=d_c+10.0*acc_feat[cls*7+1];inb_f=(n_fg_vis*(1.0/64.0))/(stats12[1]if stats12[1]>0.05 else 0.05)
  if inb_f>1.0:inb_f=1.0
  q_sc=8.0+snap_sc;qual=p_coin*(q_sc if q_sc>1.0 else 1.0);prio=qual*(0.80+0.16*inb_f+0.04*(n_fg_vis*(1.0/64.0)));return (probs,cls,p_coin,prio,n_fg_vis,snap_sc,s_best,sim_mean)
 def _eval_polar_and_accept(self,Y,Cb,Cr,fg,claimed,cx:float,cy:float):
  st=self._extract_features(Y,Cb,Cr,fg,claimed,cx,cy,need_full=False)
  if st is None:return None
  ckey=(cx,cy);ec=self._eval_cache.get(ckey)
  if ec is not None and ec[0]is st:return ec[1]
  F,vis,F_b0,_F_full,V_b0,V_full,stats12,df0,df1,denF,n_fg_vis,all_vis=st;f0,fc1,fc2,fc3,fc4,fc5,fc6,fc7,f8=_RFFT16_RI((*F[0],*F[1],*F[2],*F[3]));s_best=[];sim_best=[];sim_mean=[];nm_triples=[];dw_bests=[];sqrt=math.sqrt;dot=_dot;inv16=1.0/16.0;irfft16=_irfft16_unscaled
  if all_vis:
   for cls in range(3):
    Wri=self.W_fft_ri[cls];Wnri=self.W_fft_nri[cls];num_arr=irfft16(dot(f0,self.W_fft_0[cls]),dot(fc1,Wri[0]),dot(fc1,Wnri[0]),dot(fc2,Wri[1]),dot(fc2,Wnri[1]),dot(fc3,Wri[2]),dot(fc3,Wnri[2]),dot(fc4,Wri[3]),dot(fc4,Wnri[3]),dot(fc5,Wri[4]),dot(fc5,Wnri[4]),dot(fc6,Wri[5]),dot(fc6,Wnri[5]),dot(fc7,Wri[6]),dot(fc7,Wnri[6]),dot(f8,self.W_fft_8[cls]),);inv_scale=inv16/sqrt(denF*self.dw_full_all[cls][0]);bk=0;bv=-1e9;sm_tot=0.0
    for s in range(16):
     sv=num_arr[s]*inv_scale;sm_tot+=sv
     if sv>bv:bv=sv;bk=s
    s_best.append(bk);sim_best.append(bv);sim_mean.append(sm_tot*inv16);nm_triples.append((num_arr[bk]*inv16,num_arr[(bk+1)&15]*inv16,num_arr[(bk-1)&15]*inv16))
  else:
   v0,vc1,vc2,vc3,vc4,vc5,vc6,vc7,v8=_RFFT16_RI(vis)
   for cls in range(3):
    Wri=self.W_fft_ri[cls];Wnri=self.W_fft_nri[cls];W2ri=self.W2_fft_ri[cls];W2nri=self.W2_fft_nri[cls];nm_arr=irfft16(dot(f0,self.W_fft_0[cls]),dot(fc1,Wri[0]),dot(fc1,Wnri[0]),dot(fc2,Wri[1]),dot(fc2,Wnri[1]),dot(fc3,Wri[2]),dot(fc3,Wnri[2]),dot(fc4,Wri[3]),dot(fc4,Wnri[3]),dot(fc5,Wri[4]),dot(fc5,Wnri[4]),dot(fc6,Wri[5]),dot(fc6,Wnri[5]),dot(fc7,Wri[6]),dot(fc7,Wnri[6]),dot(f8,self.W_fft_8[cls]),);dw_arr=irfft16(dot(v0,self.W2_fft_0[cls]),dot(vc1,W2ri[0]),dot(vc1,W2nri[0]),dot(vc2,W2ri[1]),dot(vc2,W2nri[1]),dot(vc3,W2ri[2]),dot(vc3,W2nri[2]),dot(vc4,W2ri[3]),dot(vc4,W2nri[3]),dot(vc5,W2ri[4]),dot(vc5,W2nri[4]),dot(vc6,W2ri[5]),dot(vc6,W2nri[5]),dot(vc7,W2ri[6]),dot(vc7,W2nri[6]),dot(v8,self.W2_fft_8[cls]),);bk=0;bv=-1e9;sm_tot=0.0;bdw=1e-5
    for s in range(16):
     nm=nm_arr[s]*inv16;dw=dw_arr[s]*inv16+1e-5
     if dw<1e-5:dw=1e-5
     sv=nm/sqrt(denF*dw);sm_tot+=sv
     if sv>bv:bv=sv;bk=s;bdw=dw
    s_best.append(bk);sim_best.append(bv);sim_mean.append(sm_tot*inv16);dw_bests.append(bdw);nm_triples.append((nm_arr[bk]*inv16,nm_arr[(bk+1)&15]*inv16,nm_arr[(bk-1)&15]*inv16))
  res=self._finish_accept(F_b0,V_b0,V_full,stats12,df0,df1,denF,n_fg_vis,s_best,sim_best,sim_mean,nm_triples,dw_bests,all_vis);self._eval_cache[ckey]=(st,res);return res
 def _eval_polar_window(self,Y,Cb,Cr,fg,claimed,cx:float,cy:float,s_hint,sim_mean_hint):
  st=self._extract_features(Y,Cb,Cr,fg,claimed,cx,cy,need_full=True)
  if st is None:return None
  _,_,F_b0,F_full,V_b0,V_full,stats12,df0,df1,denF,n_fg_vis,all_vis=st;sqrt=math.sqrt;dot=_dot;s_best=[];sim_best=[];nm_triples=[];dw_bests=[];Wf=self.W_full;W2f=self.W2_full
  for k in range(3):
   sh=s_hint[k];sm=(sh-1)&15;sp=(sh+1)&15;Wfk=Wf[k];nm_m=dot(F_full,Wfk[sm]);nm_0=dot(F_full,Wfk[sh]);nm_p=dot(F_full,Wfk[sp])
   if all_vis:inv_d=1.0/sqrt(denF*self.dw_full_all[k][0]);sv_m=nm_m*inv_d;sv_0=nm_0*inv_d;sv_p=nm_p*inv_d;dw_m=dw_0=dw_p=0.0
   else:W2fk=W2f[k];dw_m=dot(V_full,W2fk[sm])+1e-5;sv_m=nm_m/sqrt(denF*dw_m);dw_0=dot(V_full,W2fk[sh])+1e-5;sv_0=nm_0/sqrt(denF*dw_0);dw_p=dot(V_full,W2fk[sp])+1e-5;sv_p=nm_p/sqrt(denF*dw_p)
   if sv_0>=sv_m and sv_0>=sv_p:bk=sh;bv=sv_0;bdw=dw_0;nm_triples.append((nm_0,nm_p,nm_m))
   elif sv_p>=sv_m:bk=sp;bv=sv_p;bdw=dw_p;nm_triples.append((nm_p,dot(F_full,Wfk[(sp+1)&15]),nm_0))
   else:bk=sm;bv=sv_m;bdw=dw_m;nm_triples.append((nm_m,nm_0,dot(F_full,Wfk[(sm-1)&15])))
   s_best.append(bk);sim_best.append(bv);dw_bests.append(bdw)
  return self._finish_accept(F_b0,V_b0,V_full,stats12,df0,df1,denF,n_fg_vis,s_best,sim_best,sim_mean_hint,nm_triples,dw_bests,all_vis)
 def _quick_snap_1px(self,Y,Cb,Cr,fg,claimed,cx:float,cy:float,best_res,allow_far:bool=False):
  best=None;best_sc=best_res[5];min_p=0.06 if best_res[2]<0.06 else 0.50;use_win=best_res[2]>=0.25
  if use_win:
   for dx,dy in ((1.0,0.0),(-1.0,0.0),(0.0,1.0),(0.0,-1.0)):
    nx,ny=cx+dx,cy+dy
    if nx<-20.0 or nx>520.0 or ny<-20.0 or ny>520.0:continue
    r=self._eval_polar_window(Y,Cb,Cr,fg,claimed,nx,ny,best_res[6],best_res[7])
    if r is not None and r[2]>=min_p and r[5]>best_sc:best_sc=r[5];best=(nx,ny,r)
  else:
   cand_ax=[]
   for dx,dy in ((1.0,0.0),(-1.0,0.0),(0.0,1.0),(0.0,-1.0)):
    nx,ny=cx+dx,cy+dy
    if nx<-20.0 or nx>520.0 or ny<-20.0 or ny>520.0:continue
    ncp,nfc=self._coarse_score(Y,Cb,Cr,fg,claimed,nx,ny)
    if nfc>=4 and ncp>=0.04:cand_ax.append((ncp,nx,ny))
   if len(cand_ax)>1:cand_ax.sort(reverse=True)
   for idx in range(min(2,len(cand_ax))):
    _,nx,ny=cand_ax[idx];r=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,nx,ny)
    if r is not None and r[2]>=min_p and r[5]>best_sc:
     best_sc=r[5];best=(nx,ny,r)
     if r[2]>=0.85:break
  if allow_far and best is None:
   for dx,dy in ((2.0,0.0),(-2.0,0.0),(0.0,2.0),(0.0,-2.0)):
    nx,ny=cx+dx,cy+dy
    if nx<-20.0 or nx>520.0 or ny<-20.0 or ny>520.0:continue
    ncp,nfc=self._coarse_score(Y,Cb,Cr,fg,claimed,nx,ny)
    if nfc<4 or ncp<0.04:continue
    r=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,nx,ny)
    if r is not None and r[2]>=min_p and r[5]>best_sc:best_sc=r[5];best=(nx,ny,r)
  if best is None and (allow_far or best_res[2]>=0.06):
   cand_d=[]
   for dx,dy in ((1.0,1.0),(1.0,-1.0),(-1.0,1.0),(-1.0,-1.0)):
    nx,ny=cx+dx,cy+dy
    if nx<-20.0 or nx>520.0 or ny<-20.0 or ny>520.0:continue
    ncp,nfc=self._coarse_score(Y,Cb,Cr,fg,claimed,nx,ny)
    if nfc>=4 and ncp>=0.04:cand_d.append((ncp,nx,ny))
   if len(cand_d)>1:cand_d.sort(reverse=True)
   for idx in range(len(cand_d)if allow_far else min(2,len(cand_d))):
    _,nx,ny=cand_d[idx];r=self._eval_polar_window(Y,Cb,Cr,fg,claimed,nx,ny,best_res[6],best_res[7])if use_win else self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,nx,ny)
    if r is not None and r[2]>=min_p and r[5]>best_sc:
     best_sc=r[5];best=(nx,ny,r)
     if r[2]>=0.85:break
  if allow_far and best is not None and best[2][2]<0.72:
   bx,by,bres=best
   for dx,dy in ((1.0,0.0),(-1.0,0.0),(0.0,1.0),(0.0,-1.0)):
    nx,ny=bx+dx,by+dy
    if (nx==cx and ny==cy)or nx<-20.0 or nx>520.0 or ny<-20.0 or ny>520.0:continue
    r=self._eval_polar_window(Y,Cb,Cr,fg,claimed,nx,ny,bres[6],bres[7])
    if r is not None and r[2]>=min_p and r[5]>best_sc:best_sc=r[5];best=(nx,ny,r)
  if best is not None and use_win and (best[2][2]<0.96 or max(best[2][0][1:])<0.96):
   exact=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,best[0],best[1])
   if exact is not None:best=(best[0],best[1],exact)
  return best
 def _refine_center(self,Y,Cb,Cr,fg,claimed,cx:float,cy:float,anchor_res=None,coarse:bool=False):
  best_res=anchor_res if anchor_res is not None else self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,cx,cy);best_x,best_y=cx,cy
  if best_res is None:return None
  if (not coarse)and anchor_res is not None and best_res[2]>=0.985 and max(best_res[0][1:])>=0.985 and best_res[4]>=48 and best_res[3]>=18.0:return best_x,best_y,best_res
  moved=False;used_win=False
  if coarse or best_res[2]<0.75:steps=(2.0,1.0,0.5);max_it=2
  elif best_res[2]<0.92:steps=(1.0,0.5);max_it=2
  else:steps=(0.5,);max_it=1
  for step in steps:
   skip_dx=0.0;skip_dy=0.0
   for _it in range(max_it):
    step_best_res=best_res;step_best_x=best_x;step_best_y=best_y;step_used_win=used_win;win_dx=0.0;win_dy=0.0;improved=False;use_full=(best_res[2]<0.16);min_sc=-1e9 if (coarse or best_res[2]>=0.05)else -6.0
    for dx,dy in ((step,0.0),(-step,0.0),(0.0,step),(0.0,-step)):
     if dx==skip_dx and dy==skip_dy:continue
     nx,ny=best_x+dx,best_y+dy
     if nx<-20.0 or nx>520.0 or ny<-20.0 or ny>520.0:continue
     r=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,nx,ny)if use_full else self._eval_polar_window(Y,Cb,Cr,fg,claimed,nx,ny,best_res[6],best_res[7])
     if r is not None and r[5]>step_best_res[5]+1e-3 and (r[5]>min_sc or r[5]>best_res[5]+2.5):step_best_res=r;step_best_x=nx;step_best_y=ny;step_used_win=not use_full;win_dx=dx;win_dy=dy;improved=True
    if not improved:break
    best_res=step_best_res;best_x=step_best_x;best_y=step_best_y;used_win=step_used_win;skip_dx=-win_dx;skip_dy=-win_dy;moved=True
   if coarse and ((step==2.0 and best_res[2]<0.08)or (step==1.0 and best_res[2]<0.25)):break
  if moved and used_win and (best_res[2]<0.96 or max(best_res[0][1:])<0.96):
   exact=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,best_x,best_y)
   if exact is not None:best_res=exact
  return best_x,best_y,best_res
 def _find_unclaimed_blobs(self,fg_rows,claimed_rows,min_pts:int=32):
  rem=[fg_rows[y]&~claimed_rows[y]for y in range(500)];blobs=[];popcnt=_popcnt
  for sy in range(500):
   row_m=rem[sy]
   while row_m:
    lsb=row_m&-row_m;stop=(row_m+lsb)&~row_m;run0=stop-lsb;rem[sy]^=run0;row_m^=run0;comp_rows={sy:run0};pts_cnt=popcnt(run0);q=[(sy,run0)]
    while q:
     y,m=q.pop()
     for ny in (y-1,y+1):
      if 0<=ny<500:
       nr=rem[ny];seed=nr&m
       while seed:s0=seed&-seed;lz=(~nr)&(s0-1);rs=(1<<lz.bit_length())if lz else 1;st=(nr+rs)&~nr;run_mask=st-rs;rem[ny]^=run_mask;comp_rows[ny]=comp_rows.get(ny,0)|run_mask;pts_cnt+=popcnt(run_mask);seed&=~run_mask;q.append((ny,run_mask))
    if pts_cnt>=min_pts:blobs.append((pts_cnt,comp_rows))
  blobs.sort(key=lambda t:-t[0]);return blobs
 def _scan_blob_candidates(self,Y,Cb,Cr,fg,claimed,uncl_rows,uncl_cnt:int,accepted,acc_grid,min_p:float=0.50,early_p:float=0.88,min_nms_sq:float=9.0,min_cp:float=0.08,max_nms:int=8,early_prio:float=0.0,max_ret:int=1,allow_fallback:bool=True,cs_cache=None,ref_cache=None,deadline=None,):
  row_or=0
  for um in uncl_rows.values():row_or|=um
  bx0=max(-18,((row_or&-row_or).bit_length()-1)-18);bx1=min(518,(row_or.bit_length()-1)+19);by0=max(-18,min(uncl_rows)-18);by1=min(518,max(uncl_rows)+19);area=(bx1-bx0)*(by1-by0);ag=acc_grid;m39=_MASK39
  def _probe_grid(stride:int,nms_sq:float,lim_nms:int,cur_cands):
   coarse_hits=[]
   for gy in range(by0,by1,stride):
    if deadline is not None and time.perf_counter()>deadline:return cur_cands
    local_mask=0;ry_a=0 if gy<19 else gy-19;ry_b=500 if gy>480 else gy+20
    for ry in range(ry_a,ry_b):
     um=uncl_rows.get(ry)
     if um:local_mask|=um
    if not local_mask:continue
    fgy=float(gy);cy_g=(gy+32)>>3
    if cy_g<1:cy_g=1
    elif cy_g>70:cy_g=70
    rm_g=(cy_g-1)*72;r0_g=cy_g*72;rp_g=(cy_g+1)*72;rx0=((local_mask&-local_mask).bit_length()-1)-19;rx1=(local_mask.bit_length()-1)+20
    if rx0>bx0:gx_start=bx0+((rx0-bx0+stride-1)//stride)*stride
    else:gx_start=bx0
    gx_end=rx1 if rx1<bx1 else bx1
    for gx in range(gx_start,gx_end,stride):
     if gx<=19:
      if not (local_mask&((1<<(gx+20))-1)):continue
     elif not ((local_mask>>(gx-19))&m39):continue
     fgx=float(gx);cx_g=(gx+32)>>3
     if cx_g<1:cx_g=1
     elif cx_g>70:cx_g=70
     near_a=False
     for rb in (rm_g+cx_g,r0_g+cx_g,rp_g+cx_g):
      for oi in (rb-1,rb,rb+1):
       lst=ag[oi]
       if lst is not None:
        for ax,ay in lst:
         dx=fgx-ax;dy=fgy-ay
         if dx*dx+dy*dy<56.25:near_a=True;break
        if near_a:break
      if near_a:break
     if near_a:continue
     ckey=(gx,gy)
     if cs_cache is not None and ckey in cs_cache:cp_v,fc=cs_cache[ckey]
     else:
      cp_v,fc=self._coarse_score(Y,Cb,Cr,fg,claimed,fgx,fgy)
      if cs_cache is not None:cs_cache[ckey]=(cp_v,fc)
     if cp_v>=min_cp and fc>=4:coarse_hits.append((fgx,fgy,cp_v))
   coarse_hits.sort(key=lambda t:-t[2]);nms_hits=[]
   for gx,gy,sc in coarse_hits:
    ok=True
    for hx,hy,_ in nms_hits:
     if (gx-hx)*(gx-hx)+(gy-hy)*(gy-hy)<nms_sq:ok=False;break
    if ok:
     nms_hits.append((gx,gy,sc))
     if len(nms_hits)>=lim_nms:break
   for hi,(gx,gy,sc)in enumerate(nms_hits):
    if deadline is not None and time.perf_counter()>deadline:break
    rkey=(gx,gy,sc>=0.90)
    if ref_cache is not None and rkey in ref_cache:ref=ref_cache[rkey]
    else:
     ref=self._refine_center(Y,Cb,Cr,fg,claimed,gx,gy,coarse=True)
     if (ref is None or ref[2][2]<0.25)and sc>=0.90:
      for ddx,ddy in ((-0.5,-0.5),(0.5,-0.5),(-0.5,0.5),(0.5,0.5)):
       nx,ny=gx+ddx,gy+ddy
       if nx<-20.0 or nx>520.0 or ny<-20.0 or ny>520.0:continue
       sub=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,nx,ny)
       if sub is not None and sub[2]>=0.30:
        sref=self._refine_center(Y,Cb,Cr,fg,claimed,nx,ny,anchor_res=sub,coarse=True)
        if sref is not None and (ref is None or sref[2][5]>ref[2][5]):ref=sref
     if ref_cache is not None:ref_cache[rkey]=ref
    if ref is not None and ref[2][2]>=min_p:
     rx_r,ry_r=ref[0],ref[1]
     if _near_acc(ag,rx_r,ry_r,9.0):continue
     dup_idx=-1
     for ci,c in enumerate(cur_cands):
      if (rx_r-c[0])*(rx_r-c[0])+(ry_r-c[1])*(ry_r-c[1])<9.0:dup_idx=ci;break
     if dup_idx>=0:
      if ref[2][3]>cur_cands[dup_idx][2][3]:cur_cands[dup_idx]=ref
     else:cur_cands.append(ref)
    if max_ret==1 and cur_cands and cur_cands[0][2][2]>=(0.96 if hi==0 else early_p)and cur_cands[0][2][3]>=early_prio:break
    if max_ret>1 and len(cur_cands)>=max_ret:break
   return cur_cands
  stride=3 if area<=4200 else 4;cands=_probe_grid(stride,min_nms_sq,max_nms,[]);cands.sort(key=lambda r:-r[2][3])
  if allow_fallback and uncl_cnt>=120 and area<=6000 and not cands:cands=_probe_grid(2,max(min_nms_sq,14.0),max(max_nms,12),cands);cands.sort(key=lambda r:-r[2][3])
  return cands[:max_ret]
 def _mark_claimed(self,claimed,claimed_rows,cx:float,cy:float,r:float):
  r2=r*r;y0=max(0,int(cy-r-1));y1=min(500,int(cy+r+2));sqrt=math.sqrt;ceil=math.ceil;floor=math.floor;ones=_ONES_TBL
  for y in range(y0,y1):
   rem=r2-(y-cy)*(y-cy)
   if rem>=0.0:
    dxm=sqrt(rem);lx=ceil(cx-dxm)
    if lx<0:lx=0
    else:
     while lx>0 and (lx-1-cx)*(lx-1-cx)<=rem:lx-=1
    rx=floor(cx+dxm)
    if rx>499:rx=499
    else:
     while rx<499 and (rx+1-cx)*(rx+1-cx)<=rem:rx+=1
    while lx<=rx and (lx-cx)*(lx-cx)>rem:lx+=1
    while rx>=lx and (rx-cx)*(rx-cx)>rem:rx-=1
    w=rx-lx+1
    if w>0:claimed_rows[y]|=((1<<w)-1)<<lx;off=y*500;claimed[off+lx:off+rx+1]=ones[w]if w<48 else (b"\x01"*w)
 def _coarse_snap_2px(self,Y,Cb,Cr,fg,claimed,cx:float,cy:float,cp_base:float):
  best_cp=cp_base+0.04;bx,by=cx,cy;moved=False
  for dx,dy in ((2.0,0.0),(-2.0,0.0),(0.0,2.0),(0.0,-2.0)):
   nx,ny=cx+dx,cy+dy
   if nx<-18.0 or nx>518.0 or ny<-18.0 or ny>518.0:continue
   ncp,nfc=self._coarse_score(Y,Cb,Cr,fg,claimed,nx,ny)
   if nfc>=4 and ncp>best_cp:best_cp=ncp;bx,by=nx,ny;moved=True
  return (bx,by)if moved else None
 def _refresh_pool_item(self,Y,Cb,Cr,fg,claimed,item,add_dorm)->bool:
  prev_res=item[3];was_centered=item[5]
  if prev_res is not None and prev_res[2]>=0.50:
   res=self._eval_polar_window(Y,Cb,Cr,fg,claimed,item[0],item[1],prev_res[6],prev_res[7])
   if res is not None and (res[2]<0.85 or res[4]<prev_res[4]*0.75):res=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,item[0],item[1])
  else:res=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,item[0],item[1])
  if res is not None and res[2]<0.06 and res[4]>=4:
   ix,iy=item[0],item[1];is_b=ix<24.0 or ix>476.0 or iy<24.0 or iy>476.0
   if is_b or (res[4]>=10 and item[4]<=3):
    cpd,fc_d=self._coarse_score(Y,Cb,Cr,fg,claimed,ix,iy)
    if fc_d>=4 and cpd>=(0.30 if is_b else 0.60):
     snap=self._quick_snap_1px(Y,Cb,Cr,fg,claimed,ix,iy,res,allow_far=is_b)
     if snap is not None:item[0],item[1],res=snap;was_centered=False
     elif cpd>=0.30:
      pos2=self._coarse_snap_2px(Y,Cb,Cr,fg,claimed,ix,iy,cpd)
      if pos2 is not None:
       r2=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,pos2[0],pos2[1])
       if r2 is not None and r2[2]>=0.06:item[0],item[1],res=pos2[0],pos2[1],r2;was_centered=False
  item[3]=res;item[2]=False;item[5]=bool(was_centered and prev_res is not None and res is not None and res[4]==prev_res[4]and res[2]>=prev_res[2]-0.02)
  if res is None or res[2]<0.06:
   if res is not None and res[4]>=3 and item[4]<16:add_dorm(item[0],item[1],item[4]+1)
   return False
  return True
 def solve(self,jpeg_bytes:bytes,time_budget_s:float=1.80)->int:
  was_gc=gc.isenabled();gc.disable()
  try:
   self._pt_cache={};self._eval_cache={};t_start=time.perf_counter();deadline=t_start+time_budget_s;Y,Cb,Cr=decode_jpeg_half(jpeg_bytes,deadline=t_start+time_budget_s*0.68)
   if isinstance(Y,(bytes,bytearray))and len(Y)==250000 and len(Cb)==250000 and len(Cr)==250000:yv=int.from_bytes(Y,"big");cbv=int.from_bytes(Cb,"big");crv=int.from_bytes(Cr,"big");ym=yv&(yv>>1);y_lt240=(~((ym&(ym>>2))>>4))&_SWAR_01;icb=((~cbv)&_SWAR_80)>>7;acb=((cbv&_SWAR_7F)^((icb<<7)-icb))+icb;icr=((~crv)&_SWAR_80)>>7;acr=((crv&_SWAR_7F)^((icr<<7)-icr))+icr;big=(acb|acr)&_SWAR_F0;lt11=(((acb&_SWAR_0F)+(acr&_SWAR_0F)+_SWAR_117)&_SWAR_80)>>7;fg_bytes=(y_lt240|lt11|(((big|((big&_SWAR_7F)+_SWAR_70))&_SWAR_80)>>7)).to_bytes(250000,"big");fg=bytearray(fg_bytes)
   else:_abs=abs;fg=bytearray(1 if (y<240.0 or _abs(cb-128.0)+_abs(cr-128.0)>10.0)else 0 for y,cb,cr in zip(Y,Cb,Cr));fg_bytes=bytes(fg)
   big_bits=int(fg_bytes[::-1].translate(_TRANS_FG),2);fg_rows=[(big_bits>>(y*500))&_MASK500 for y in range(500)];del fg_bytes,big_bits;claimed=bytearray(250000);claimed_rows=[0]*500;raw_cands=self._propose_centers(Y,Cb,Cr,fg,fg_rows=fg_rows,deadline=t_start+time_budget_s*0.78);raw_cands.sort(key=lambda t:-t[2]);dedup_cands=[];dd_grid=[None]*(144*144)
   for cx,cy,votes in raw_cands:
    gx=int(cx+24.0)>>2;gy=int(cy+24.0)>>2;keep=True
    for ky in (gy-1,gy,gy+1):
     ry=ky*144
     for kx in (gx-1,gx,gx+1):
      cell=dd_grid[ry+kx]
      if cell is not None:
       for px,py in cell:
        if (cx-px)*(cx-px)+(cy-py)*(cy-py)<9.0:keep=False;break
      if not keep:break
     if not keep:break
    if keep:
     dedup_cands.append((cx,cy,votes));cidx=gy*144+gx;cell=dd_grid[cidx]
     if cell is None:dd_grid[cidx]=[(cx,cy)]
     else:cell.append((cx,cy))
   dorm_grid=[None]*(18*18)
   def add_dorm(dx_c:float,dy_c:float,w:int):
    gx=int(dx_c+32.0)>>5;gy=int(dy_c+32.0)>>5
    if gx<0:gx=0
    elif gx>17:gx=17
    if gy<0:gy=0
    elif gy>17:gy=17
    ci=gy*18+gx;cell=dorm_grid[ci]
    if cell is None:dorm_grid[ci]=[[dx_c,dy_c,w]]
    else:cell.append([dx_c,dy_c,w])
   pool=[];cand_dl=deadline-0.16
   for _ci,(cx,cy,votes)in enumerate(dedup_cands):
    if (_ci&7)==7 and time.perf_counter()>cand_dl:break
    is_border=cx<24.0 or cx>476.0 or cy<24.0 or cy>476.0;cp_val,fg_cnt=self._coarse_score(Y,Cb,Cr,fg,claimed,cx,cy);sx,sy=cx,cy;shifted_coarse=False
    if (cp_val<0.06 or fg_cnt<4)and ((is_border and votes>=1.3)or votes>=4.5):
     best_cp=cp_val if fg_cnt>=4 else 0.0
     for dx,dy in ((2.0,0.0),(-2.0,0.0),(0.0,2.0),(0.0,-2.0)):
      nx,ny=cx+dx,cy+dy
      if nx<-18.0 or nx>518.0 or ny<-18.0 or ny>518.0:continue
      ncp,nfc=self._coarse_score(Y,Cb,Cr,fg,claimed,nx,ny)
      if nfc>=4 and ncp>best_cp:best_cp=ncp;sx,sy=nx,ny;fg_cnt=nfc;cp_val=best_cp;shifted_coarse=True
    if cp_val>=0.06 and fg_cnt>=4:
     res=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,sx,sy)
     if (res is None or res[2]<0.06)and cp_val>=0.30:
      pos2=self._coarse_snap_2px(Y,Cb,Cr,fg,claimed,sx,sy,cp_val)
      if pos2 is not None:sx,sy=pos2;shifted_coarse=True;res=self._eval_polar_and_accept(Y,Cb,Cr,fg,claimed,sx,sy)
     if shifted_coarse:add_dorm(cx,cy,0)
     if res is not None and res[2]<0.45 and res[4]>=4 and (res[2]>=0.06 or is_border):
      snap=self._quick_snap_1px(Y,Cb,Cr,fg,claimed,sx,sy,res,allow_far=is_border)
      if snap is not None:sx,sy,res=snap
     if res is not None and res[2]>=0.06:pool.append([sx,sy,False,res,0,False])
     elif not shifted_coarse:add_dorm(cx,cy,0)
    elif fg_cnt>=4:add_dorm(cx,cy,0)
   accepted=[];acc_grid=[None]*(72*72);refresh_item=self._refresh_pool_item;pool_dl=deadline-0.03;_sc=0;has_dirty=False;has_none=False
   while pool:
    _sc+=1
    if (_sc&3)==0 and time.perf_counter()>pool_dl:break
    if has_none:
     has_none=False;new_pool=[]
     for it in pool:
      if it[3]is None:
       if not refresh_item(Y,Cb,Cr,fg,claimed,it,add_dorm):continue
      new_pool.append(it)
     pool=new_pool
     if not pool:break
    if has_dirty:
     while pool:
      if time.perf_counter()>pool_dl:break
      best_idx=0;best_prio=pool[0][3][3]
      for idx in range(1,len(pool)):
       pr=pool[idx][3][3]
       if pr>best_prio:best_prio=pr;best_idx=idx
      bx,by=pool[best_idx][0],pool[best_idx][1];refreshed_any=False;dropped_any=False;any_dirty_left=False
      for idx,item in enumerate(pool):
       if item[2]:
        if idx==best_idx or (item[0]-bx)*(item[0]-bx)+(item[1]-by)*(item[1]-by)<=1521.0:
         if time.perf_counter()>pool_dl:any_dirty_left=True;break
         refreshed_any=True
         if not refresh_item(Y,Cb,Cr,fg,claimed,item,add_dorm):pool[idx]=None;dropped_any=True
        else:any_dirty_left=True
      if dropped_any:pool=[it for it in pool if it is not None]
      if not refreshed_any:has_dirty=any_dirty_left;break
     if not pool or time.perf_counter()>pool_dl:break
    best_idx=0;best_prio=pool[0][3][3]
    for idx in range(1,len(pool)):
     pr=pool[idx][3][3]
     if pr>best_prio:best_prio=pr;best_idx=idx
    cx,cy,_,res,wakes,centered=pool.pop(best_idx)
    if centered:rcx,rcy,rres=cx,cy,res
    else:
     ref=self._refine_center(Y,Cb,Cr,fg,claimed,cx,cy,anchor_res=res)
     if ref is None:continue
     rcx,rcy,rres=ref
    probs,cls,p_coin,prio,n_fg_vis=rres[0],rres[1],rres[2],rres[3],rres[4]
    if p_coin<0.45 or n_fg_vis<4:
     if n_fg_vis>=3 and wakes<16:add_dorm(cx,cy,wakes+1)
     continue
    if _near_acc(acc_grid,rcx,rcy,9.0):continue
    beaten=False;dropped_any=False
    for idx,it in enumerate(pool):
     ix,iy=it[0],it[1];dx=ix-rcx;dy=iy-rcy;d2=dx*dx+dy*dy
     if d2>1600.0:continue
     if it[2]:
      if not refresh_item(Y,Cb,Cr,fg,claimed,it,add_dorm):pool[idx]=None;dropped_any=True;continue
      ix,iy=it[0],it[1];dx=ix-rcx;dy=iy-rcy;d2=dx*dx+dy*dy
     ires=it[3];is_ib=ix<24.0 or ix>476.0 or iy<24.0 or iy>476.0
     if (not it[5])and d2<=1444.0 and (ires[2]>=0.25 or is_ib)and (ires[3]>=prio-4.5 or (is_ib and ires[3]>=prio-7.5))and (ires[2]<0.90 or ires[3]>=prio-2.0):
      iw=it[4];iref=self._refine_center(Y,Cb,Cr,fg,claimed,ix,iy,anchor_res=ires)
      if iref is None:pool[idx]=None;dropped_any=True;continue
      nrx,nry,nres=iref
      if nres[2]<0.06:
       if nres[4]>=3 and iw<16:add_dorm(ix,iy,iw+1)
       pool[idx]=None;dropped_any=True;continue
      it[0],it[1],it[3],it[5]=nrx,nry,nres,True
      if nres[2]>=0.45 and nres[4]>=4 and nres[3]>prio+0.05:beaten=True
    if dropped_any:pool=[it for it in pool if it is not None]
    if beaten:pool.append([rcx,rcy,False,rres,wakes,True]);continue
    accepted.append((cls,rcx,rcy));self._mark_claimed(claimed,claimed_rows,rcx,rcy,self.radii_500[cls]+0.12);_add_acc(acc_grid,rcx,rcy)
    for it in pool:
     dx=it[0]-rcx;dy=it[1]-rcy
     if dx*dx+dy*dy<=1521.0:it[2]=True;has_dirty=True
    cgx=int(rcx+32.0)>>5;cgy=int(rcy+32.0)>>5;gy0=0 if cgy<2 else cgy-2;gy1=17 if cgy>15 else cgy+2;gx0=0 if cgx<2 else cgx-2;gx1=17 if cgx>15 else cgx+2
    for dgy_c in range(gy0,gy1+1):
     rbase=dgy_c*18
     for dgx_c in range(gx0,gx1+1):
      cell=dorm_grid[rbase+dgx_c]
      if cell is None:continue
      new_cell=None
      for di,d_it in enumerate(cell):
       dx_c,dy_c,d_wakes=d_it
       if d_wakes<16 and (dx_c-rcx)*(dx_c-rcx)+(dy_c-rcy)*(dy_c-rcy)<=1521.0:
        if new_cell is None:new_cell=cell[:di]
        if _near_acc(acc_grid,dx_c,dy_c,25.0):continue
        cp_d,fc=self._coarse_score(Y,Cb,Cr,fg,claimed,dx_c,dy_c)
        if cp_d>=0.06 and fc>=4:pool.append([dx_c,dy_c,True,None,d_wakes+1,False]);has_none=True;has_dirty=True
        elif fc>=4:d_it[2]=d_wakes+1;new_cell.append(d_it)
       elif new_cell is not None:new_cell.append(d_it)
      if new_cell is not None:dorm_grid[rbase+dgx_c]=new_cell if new_cell else None
   popcnt=_popcnt
   if time.perf_counter()<deadline-0.03:
    for _cnt,comp_rows in self._find_unclaimed_blobs(fg_rows,claimed_rows,min_pts=44):
     if time.perf_counter()>deadline:break
     cs_cache={};ref_cache={}
     for _sub in range(48):
      if time.perf_counter()>deadline:break
      uncl_rows={};uncl_cnt=0
      for y,m in comp_rows.items():
       um=m&~claimed_rows[y]
       if um:uncl_rows[y]=um;uncl_cnt+=popcnt(um)
      if uncl_cnt<28:break
      cands=self._scan_blob_candidates(Y,Cb,Cr,fg,claimed,uncl_rows,uncl_cnt,accepted,acc_grid,min_p=0.50,early_p=0.88,min_nms_sq=9.0,min_cp=0.14,max_nms=5,early_prio=15.5,cs_cache=cs_cache,ref_cache=ref_cache,deadline=deadline,)
      if not cands:break
      rcx,rcy,rres=cands[0];cls=rres[1];accepted.append((cls,rcx,rcy));self._mark_claimed(claimed,claimed_rows,rcx,rcy,self.radii_500[cls]+0.12);_add_acc(acc_grid,rcx,rcy)
      if cs_cache:
       for kxy in [kxy for kxy in cs_cache if (kxy[0]-rcx)**2+(kxy[1]-rcy)**2<=1600.0]:del cs_cache[kxy]
      if ref_cache:
       for rk in [rk for rk in ref_cache if (rk[0]-rcx)**2+(rk[1]-rcy)**2<=2304.0]:del ref_cache[rk]
   values=(1,5,10);return sum(values[c]for c,_,_ in accepted)
  finally:
   if was_gc:gc.enable()
def main():data=sys.stdin.buffer.read();weights=load_weights(WEIGHTS_B85);det=CoinDetector(weights);print(det.solve(data,time_budget_s=max(0.6,1.84-(time.perf_counter()-_T0))))
if __name__=="__main__":main()
'''


def _iter_floats(obj):
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_floats(x)


def _validate_b85(b85: str) -> str:
    b85 = b85.strip()
    if not b85:
        raise ValueError("WEIGHTS_B85 string is empty.")
    if any(ch in b85 for ch in ('"', "'", "\\", "\n", "\r", " ", "\t")):
        raise ValueError("WEIGHTS_B85 contains invalid characters or whitespace.")
    blob = base64.b85decode(b85.encode("ascii"))
    if len(blob) < 166:
        raise ValueError(f"Decoded weight blob too small ({len(blob)} bytes).")
    n_tot, len_se, len_m8 = struct.unpack_from("<HHH", blob, 0)
    expected_min = 166 + len_se + len_m8 + (n_tot + 3) // 4
    if len(blob) < expected_min:
        raise ValueError(f"Truncated weight blob: {len(blob)} < {expected_min} bytes.")
    return b85


def _load_solution_module(path: str):
    spec = importlib.util.spec_from_file_location("_generated_solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_solution(
    weights_b85: str = None,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    out_path: str = DEFAULT_SOLUTION_PATH,
    verify: bool = True,
) -> str:
    if weights_b85 is None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weight file not found: {weights_path}")
        with open(weights_path, "r", encoding="ascii") as f:
            weights_b85 = f.read()

    b85 = _validate_b85(weights_b85)
    assert "__WEIGHTS_B85__" in SOLUTION_TEMPLATE
    src = SOLUTION_TEMPLATE.replace("__WEIGHTS_B85__", b85, 1)

    # Syntax check before writing
    compile(src, out_path, "exec")

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    os.replace(tmp_path, out_path)

    n_bytes = len(src.encode("utf-8"))
    print(f"Generated {out_path} ({n_bytes} bytes total, weights={len(b85)} chars)", flush=True)
    if n_bytes > SOFT_LIMIT_BYTES:
        print(f"WARNING: {out_path} exceeds {SOFT_LIMIT_BYTES} bytes!", flush=True)

    if verify:
        t0 = time.perf_counter()
        mod = _load_solution_module(out_path)
        weights = mod.load_weights(mod.WEIGHTS_B85)
        for name, val in weights.items():
            for fv in _iter_floats(val):
                if not math.isfinite(fv):
                    raise RuntimeError(f"Non-finite weight detected in '{name}': {fv}")
        _det = mod.CoinDetector(weights)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"Verified load_weights + CoinDetector init OK in {dt_ms:.1f} ms "
            f"(radii_500={[round(r, 3) for r in weights['radii_500']]})",
            flush=True,
        )
    return out_path


def run_sample_tests(solution_path: str = DEFAULT_SOLUTION_PATH, sample_dir: str = DEFAULT_SAMPLE_DIR):
    in_files = sorted(glob.glob(os.path.join(sample_dir, "*.in")))
    if not in_files:
        print(f"No .in files found in {sample_dir}", flush=True)
        return

    mod = _load_solution_module(solution_path)
    t_init = time.perf_counter()
    weights = mod.load_weights(mod.WEIGHTS_B85)
    det = mod.CoinDetector(weights)
    init_ms = (time.perf_counter() - t_init) * 1000.0
    print(f"\nBenchmarking {solution_path} on {len(in_files)} samples (init={init_ms:.1f} ms):", flush=True)

    exact = 0
    total_abs_err = 0
    for in_path in in_files:
        name = os.path.basename(in_path)
        ans_path = os.path.splitext(in_path)[0] + ".ans"
        expected = int(open(ans_path, "r", encoding="utf-8").read().strip()) if os.path.exists(ans_path) else None
        with open(in_path, "rb") as f:
            jpg = f.read()
        t0 = time.perf_counter()
        pred = det.solve(jpg, time_budget_s=1.80)
        dt = time.perf_counter() - t0
        if expected is not None:
            err = pred - expected
            total_abs_err += abs(err)
            ok = (err == 0)
            exact += int(ok)
            status = "OK" if ok else f"DIFF {err:+d}"
            print(f"  {name:14s} pred={pred:5d}  ans={expected:5d}  [{status:10s}]  {dt:.3f}s", flush=True)
        else:
            print(f"  {name:14s} pred={pred:5d}  {dt:.3f}s", flush=True)
    print(f"Exact matches: {exact}/{len(in_files)} | Total |err| = {total_abs_err}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Generate solution.py from trained_weights.b85")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH, help="Path to trained_weights.b85")
    ap.add_argument("--out", default=DEFAULT_SOLUTION_PATH, help="Output path for solution.py")
    ap.add_argument("--no-verify", action="store_true", help="Skip load_weights/CoinDetector verification")
    ap.add_argument("--test", action="store_true", help="Run generated solution.py against ./sampleimages/*.in")
    ap.add_argument("--samples", default=DEFAULT_SAMPLE_DIR, help="Path to sampleimages directory")
    args = ap.parse_args()

    out_path = generate_solution(
        weights_path=args.weights,
        out_path=args.out,
        verify=not args.no_verify,
    )
    if args.test:
        run_sample_tests(out_path, args.samples)


if __name__ == "__main__":
    main()
