open Printf
open Stdlib
open Random
type pins = int list
type nets = string array(*pmin记录该线网的最小位置，pmax记录该线网的最大位置*)
type mos = {name:string;id:int;x:int;s:int;g:int;d:int;w:int}(*s,g,d分别是该线网源栅漏的编号，w使用nm单位时无小数，nmos、pmos分开*)

let read_cell (cells:string) (cell:string) = (*程序入口,从cells文件中找到cell标准单元*)
  let ic =
  try open_in cells
  with _ -> failwith ("reading error : " ^ cells) in
  let rec reading ic result st = 
    try let line = input_line ic in (* 读入一行 *)
    let strlst = String.split_on_char ' ' line in
    let strary = Array.of_list strlst in
    if Array.length strary>1 && strary.(0)=".SUBCKT" && strary.(1)=cell then reading ic (strlst::result) true
    else if st then 
      if Array.length strary>0 && strary.(0)=".ENDS" then List.rev (strlst::result)
      else reading ic (strlst::result) true
    else reading ic result false
    with
    End_of_file -> List.rev result
    | _ -> failwith "Preprocessing error"
in reading ic [] false

let read_cells_name cells = (*将所有标准单元的name读到lst中*)
  let ic =
    try open_in cells
    with _ -> failwith ("reading error : " ^ cells) in
    let rec reading ic result = 
      try let line = input_line ic in (* 读入一行 *)
      let strlst = String.split_on_char ' ' line in
      let strary = Array.of_list strlst in
      if Array.length strary>1 && strary.(0)=".SUBCKT" then reading ic (strary.(1)::result)
      else reading ic result
      with
      End_of_file -> List.rev result
      | _ -> failwith "Preprocessing error"
in reading ic []

(*mos_ary即为最终输出的json文件*)
let write_to_file filename str =
  let oc = open_out filename in
  output_string oc str;
  close_out oc

let write_to_file' file_name content =
  let oc = open_out_gen [Open_append; Open_creat] 0o666 file_name in
  output_string oc content;
  close_out oc

let mos_str (tp : bool) (nets : nets) (mos : mos)  = (*将mos转换成string*)
  let f tp = if tp=false then "0" else "1" in
  let str = "        \"" ^ mos.name ^ "\": {\n            \"x\": \"" ^ (string_of_int mos.x) ^ "\",\n            \"y\": \"" ^ (f tp) ^ "\",\n            \"source\": \""
  ^ nets.(mos.s) ^ "\",\n            \"gate\": \"" ^ nets.(mos.g) ^ "\",\n            \"drain\": \"" ^ nets.(mos.d) ^ "\",\n            \"width\": \""
  ^ (string_of_int mos.w) ^ "\"\n        },\n"
in str

let mos_ary_str' (tp : bool) (nets : nets) (mos_lst : mos list) = List.fold_left ( ^ ) "" (List.map (mos_str tp nets) mos_lst)
let mos_ary_str (pmos_ary : mos array) (nmos_ary : mos array) (net_ary : nets) =
  (mos_ary_str' true net_ary (Array.to_list pmos_ary)) ^ (mos_ary_str' false net_ary (Array.to_list nmos_ary))

let placement_str (pmos_ary : mos array) (nmos_ary : mos array) (net_ary : nets) = 
  let remove_last_char str =
    let len = String.length str in
    if len>2 then (String.sub str 0 (len-2)) else "" in 
"{\n    \"placement\": {\n" ^ remove_last_char (mos_ary_str pmos_ary nmos_ary net_ary) ^ "\n    }\n}\n"

let pro_head (arr:string array array) : string list =(*第一行的第3个元素开始遍历*)
  let head = arr.(0) in
  let rec f i lst = 
    if i<Array.length head then f (i+1) ((head.(i))::lst)
    else lst
in f 2 []

let pro_row (arr:string array) (lst:string list):string list =
  let rec ff j lst =
  if j<=4 then begin
  if Bool.not (List.mem arr.(j) lst) then ff (j+1) (arr.(j)::lst)
  else ff (j+1) lst end
  else lst
in ff 1 lst

let pro_middle (arr:string array array) (lst:string list):string list =(*除第一行和最后一行从第2个元素开始遍历*)
 let rec f i lst = 
  if i<(Array.length arr) -1 then f (i+1) (pro_row arr.(i) lst)
  else lst
in f 1 lst

let find_net (nets:nets) (net:string) : int = (*返回net在nets中的位置下标*)
  let rec f i =
    if i<Array.length nets then 
    if nets.(i)=net then i else f (i+1)
    else -1
in f 0

let split_width (width : int) =(*晶体管折叠，220+120(min) ~ 220+220范围内直接分成220 & else，否则用200分割*) 
  let rec f width lst = 
    if width<=440 then 
      if width mod 2=0 then 
        if (width/2) mod 5=0 then (width/2)::(width/2)::lst 
        else (width/2+3)::(width/2-2)::lst
      else (width/2+3)::(width/2-2)::lst
    else f (width-200) (200::lst)
in f width []

let order_moslst (lst:mos list) =
  let rec f lst res i = match lst with
  |(hd::tl) -> f tl ({hd with id=i}::res) (i+1)
  |[] -> List.rev res
in f lst [] 0

let init (s:string list list) : int list * string list * mos list * mos list * string * int=(*初始化数据结构,标准单元参考宽度*)
  let ss = List.map Array.of_list s in
  let arr = Array.of_list ss in
  let filename = arr.(0).(1) in
  let pins = pro_head arr in
  let nets = pro_middle arr pins in
  let nets' = Array.of_list nets in
  let pins = List.map (find_net nets') pins in(*将pin改变为保存net下标的ilist*)
  let rec f i nlst plst w_ref=
    if i<(Array.length arr) -1 then
    let r = arr.(i) in
    let name= r.(0) and s=r.(1) and g=r.(2) and d=r.(3) and tp=r.(5) and width=if String.get r.(6) 0 ='w' || String.get r.(6) 0 ='W' then r.(6) else r.(7) in
    let name = String.sub r.(0) 1 (String.length name - 1)  in
    let x = -1 in
    let s = find_net nets' s in
    let g = find_net nets' g in
    let d = find_net nets' d in
    let tp = if String.get tp 0 = 'n' then false else true in
    let float_num = float_of_string (String.sub width 2 ((String.length width)-3)) in
    let int_num = if String.get width ((String.length width)-1)='u' then int_of_float (float_num *. 1000.) else int_of_float float_num in
    let mos_split = if int_num>=240 then split_width int_num else [int_num] in
    let mk_mos_lst mos_split =
      let rec f lst res i = match lst with
      |(hd::tl) -> if i=0 then f tl ({name=name;id= -1;x=x;s=s;g=g;d=d;w=hd}::res) (i+1)
      else f tl ({name=name^"_finger"^(string_of_int i);id= -1;x=x;s=s;g=g;d=d;w=hd}::res) (i+1)
      |[] -> res
    in f mos_split [] 0 in
    let mos_list = mk_mos_lst mos_split in
      if Bool.not tp then f (i+1) (List.append mos_list nlst) plst (w_ref+((int_num+199)/220))
      else f (i+1) nlst ((List.append mos_list plst)) (w_ref+((int_num+199)/220))
    else pins,nets,order_moslst(List.rev nlst),order_moslst(List.rev plst),filename,w_ref
in f 1 [] [] 0

let convert_to_mos mos_op = match mos_op with
|Some mos-> mos
|None -> {name= "NULL";id= -1;x= -1;s= -1;g= -1;d= -1;w= 0}

let check_notch_init (mos_place : mos option array) (new_mos : mos):bool= (*检查是否出现notch，对于初始化过程来说，只需要检查前2个mos (1)均不是None (2)没有width凹槽*)
begin if Array.length mos_place <2 then false 
  else let mos1 = mos_place.((Array.length mos_place)-2) and mos2 = mos_place.((Array.length mos_place)-1) in begin
    if (mos1!=None) && (mos2!=None) then 
      if (convert_to_mos mos1).w>(convert_to_mos mos2).w && new_mos.w>(convert_to_mos mos2).w then true
      else false
    else false
  end
end

let update_mos_ary (mos_ary : mos array) (mos_place : mos option list) = (*用物理布局更新mos_ary*)
  let mos_place = Array.of_list mos_place in
  for i=0 to Array.length mos_place -1 do
    if mos_place.(i)!=None then
    let mos =convert_to_mos mos_place.(i) in
    Array.set mos_ary mos.id {mos with x=i};
  done

(*补全pmos_place和nmos_place*)
let complete_lst (lst : mos option list) (len : int) =
  let rec f none_lst = if (List.length none_lst + List.length lst)<len then f (None::none_lst)
  else List.append lst none_lst
in f []

let ary_drop ary pos = (*删除索引为pos的元素*)
  let left_ary = Array.sub ary 0 pos
  and  right_ary = Array.sub ary (pos+1) (Array.length ary-pos-1)  in
Array.append left_ary right_ary

let check_notch (place_ary:mos option array) = (*检查是否出现notch*)
  let rec f i pre_width =
    if i+1<Array.length place_ary then
      if place_ary.(i)=None then f (i+1) 0
      else if pre_width>(convert_to_mos(place_ary.(i))).w && (convert_to_mos(place_ary.(i))).w<(convert_to_mos(place_ary.(i+1))).w then false
      else f (i+1) (convert_to_mos(place_ary.(i))).w
    else true
in f 0 0

let update_place_ary (pp_ary:mos option array) (np_ary:mos option array) = (*检测是否能够删除同为None的一列*)(*注意到pp_ary和np_ary的长度一定是相同的*)
  let rec check pp_ary np_ary i =
    if i<Array.length pp_ary then
    let pmos = pp_ary.(i) and nmos = np_ary.(i) in
    let left_pmos = if i>0 then pp_ary.(i-1) else None and left_nmos = if i>0 then np_ary.(i-1) else None in
    let right_pmos = if i+1<Array.length pp_ary then pp_ary.(i+1) else None and right_nmos = if i+1<Array.length np_ary then np_ary.(i+1) else None in
      if pmos=None && nmos=None &&
      (left_pmos=None || right_pmos=None || (convert_to_mos(left_pmos)).d=(convert_to_mos(right_pmos)).s) &&
      (left_nmos=None || right_nmos=None || (convert_to_mos(left_nmos)).d=(convert_to_mos(right_nmos)).s) then
        let new_pp = (ary_drop pp_ary i) and new_np = (ary_drop np_ary i) in
        if check_notch new_pp && check_notch new_np then check new_pp new_np i
        else check pp_ary np_ary (i+1)
      else check pp_ary np_ary (i+1)
    else pp_ary,np_ary
in check pp_ary np_ary 0

(*注意到:mos_lst只有在初始时hd才为None,其他情况下hd均为Some mos,而初始时不可能出现notch,所以在源漏共用时检查notch*)
let init_layout (pmos:mos list) (nmos:mos list) = (*初始化布局，nmos,pmos合法化,并使用物理布局更新mos_ary*)
  let pmos_ary=Array.of_list pmos and nmos_ary=Array.of_list nmos in
  let mos_num = 2 * max (List.length pmos) (List.length nmos) in
  let is_empty lst = if List.length lst=0 then true else false in
  let rec place (plst:mos list) (nlst:mos list) (pp:mos option list) (np:mos option list) st =
    let pp_ary = Array.of_list (List.rev pp) and np_ary = Array.of_list (List.rev np) and pp_len = List.length pp and np_len = List.length np in
    if (Bool.not (is_empty plst)) || (Bool.not (is_empty nlst)) then
      if st then(*放置pmos*)
        let pmos = List.hd plst and pre_pmos = convert_to_mos (List.hd pp) in
        if (List.hd pp = None)||(pre_pmos.d = pmos.s) then(*可以源漏共用*)
          if pp_len >= np_len || (np_len - pp_len = 2 && (np_ary.(pp_len) = None)) || ((np_len - pp_len = 1) && pmos.g = (convert_to_mos np_ary.(pp_len)).g) then 
            if Bool.not (check_notch_init pp_ary pmos) then(*不出现notch*)
            place (List.tl plst) nlst (Some pmos::pp) np (if is_empty nlst then st else Bool.not st)
            else place (List.tl plst) nlst (Some pmos::None::pp) np (if is_empty nlst then st else Bool.not st)(*出现notch*)
          else if np_len - pp_len = 2 then place (List.tl plst) nlst (Some pmos::None::None::pp) np (if is_empty nlst then st else Bool.not st)
          else place (List.tl plst) nlst (Some pmos::None::pp) np (if is_empty nlst then st else Bool.not st)(*np_len - pp_len = 1*)
        else (*源漏无法共用*)
          if pp_len >= np_len then place (List.tl plst) nlst (Some pmos::None::pp) np (if is_empty nlst then st else Bool.not st)
          else if np_len - pp_len = 1 then place (List.tl plst) nlst (Some pmos::None::pp) np (if is_empty nlst then st else Bool.not st)
          else
            if pmos.g = (convert_to_mos (List.hd np)).g then place (List.tl plst) nlst (Some pmos::None::pp) np (if is_empty nlst then st else Bool.not st)(*np_len - pp_len = 2*)
            else place (List.tl plst) nlst (Some pmos::None::None::pp) np (if is_empty nlst then st else Bool.not st)
      else (*放置nmos*)
        let nmos = List.hd nlst and pre_nmos = convert_to_mos (List.hd np) in
        if (List.hd np = None)||(pre_nmos.d = nmos.s) then(*可以源漏共用*)
          if np_len >= pp_len || (pp_len - np_len = 2 && (pp_ary.(np_len) = None)) || ((pp_len - np_len = 1) && nmos.g = (convert_to_mos pp_ary.(np_len)).g) then 
            if Bool.not (check_notch_init np_ary nmos) then(*不出现notch*)
            place plst (List.tl nlst) pp (Some nmos::np) (if is_empty plst then st else Bool.not st)
            else place plst (List.tl nlst) pp (Some nmos::None::np) (if is_empty plst then st else Bool.not st)(*出现notch*)
          else if pp_len - np_len = 2 then place plst (List.tl nlst) pp (Some nmos::None::None::np) (if is_empty plst then st else Bool.not st)
          else place plst (List.tl nlst) pp (Some nmos::None::np) (if is_empty plst then st else Bool.not st)(*pp_len - np_len = 1*)
        else (*源漏无法共用*)
          if np_len >= pp_len then place plst (List.tl nlst) pp (Some nmos::None::np) (if is_empty plst then st else Bool.not st)
          else if pp_len - np_len = 1 then place plst (List.tl nlst) pp (Some nmos::None::np) (if is_empty plst then st else Bool.not st)
          else (*pp_len - np_len = 2*)
            if nmos.g = (convert_to_mos (List.hd pp)).g then place plst (List.tl nlst) pp (Some nmos::None::np) (if is_empty plst then st else Bool.not st)(*np_len - pp_len = 2*)
            else place plst (List.tl nlst) pp (Some nmos::None::None::np) (if is_empty plst then st else Bool.not st)
    else begin
    let pmos_place = complete_lst (List.tl(List.rev pp)) mos_num and nmos_place = complete_lst (List.tl(List.rev np)) mos_num in
    let pplace_ary,nplace_ary = update_place_ary (Array.of_list pmos_place) (Array.of_list nmos_place) in(*删除所有可删除的None组合*)
    update_mos_ary pmos_ary pmos_place;(*用物理布局更新mos_ary*)
    update_mos_ary nmos_ary nmos_place;
    pmos_ary,nmos_ary,pplace_ary,nplace_ary
    end
in place pmos nmos [None] [None] true

type pair = {pmin : float ; pmax : float}
let routing (net_ary : nets) (pp_ary : mos option array) (np_ary : mos option array):float = (*布线复杂度*)
  let nets_routing = Array.make (Array.length net_ary) {pmin=max_float;pmax=min_float} in
  let traverse ary =
    for i=0 to Array.length ary-1 do
      if ary.(i)!=None then
      let s = (convert_to_mos(ary.(i))).s and g = (convert_to_mos(ary.(i))).g and d = (convert_to_mos(ary.(i))).d in
      Array.set nets_routing s {pmin=min nets_routing.(s).pmin ((float_of_int i)-.0.5);pmax=max nets_routing.(s).pmax ((float_of_int i)-.0.5)};
      Array.set nets_routing g {pmin=min nets_routing.(g).pmin (float_of_int i);pmax=max nets_routing.(g).pmax (float_of_int i)};
      Array.set nets_routing d {pmin=min nets_routing.(d).pmin ((float_of_int i)+.0.5);pmax=max nets_routing.(d).pmax ((float_of_int i)+.0.5)};
    done; in
  let () = traverse pp_ary in
  let () = traverse np_ary in
  let rec calc_route i res =  (*计算除VSS和VDD之外的route*)
  if i<Array.length nets_routing then 
    if net_ary.(i)="VDD" || net_ary.(i)="VSS" then calc_route (i+1) res
    else calc_route (i+1) (res +. nets_routing.(i).pmax -. nets_routing.(i).pmin)
  else res
in calc_route 0 0.

let calc_symmetric (pp_ary:mos option array) (np_ary:mos option array) =(*计算栅极未配对的mos管个数*)
  let rec f i res = if i<Array.length pp_ary then
  if (pp_ary.(i)=None && np_ary.(i)!=None) || (pp_ary.(i)!=None && np_ary.(i)=None) then f (i+1) (res+1)
  else f (i+1) res
  else res
in f 0 0

let standard_deviation lst = (*计算列表的标准差*)
  let len = float_of_int (List.length lst) in
  let lst_mean = (List.fold_left (+.) 0. lst)/.len in(*平均值*)
  let f x = (x -. lst_mean) *. (x -. lst_mean) in
sqrt((List.fold_left (+.) 0. (List.map f lst))/.len)

let rec remove_duplicates lst =(*删除list中相同元素*)
  match lst with
  | [] -> []
  | hd :: tl -> hd :: (remove_duplicates (List.filter (fun x -> x <> hd) tl))

let calc_pin_access (nets:nets) (pins:int list) (pp_ary:mos option array) (np_ary:mos option array) =
  let width = float_of_int (Array.length pp_ary) in(*标准单元宽度*)
  let rec mv_pins lst res = match lst with(*从pins中删除VDD,VSS*)
  |(hd::tl) -> let name = nets.(hd) in
  if name="VSS" || name="VDD" then mv_pins tl res else mv_pins tl (hd::res)
  |[] -> List.rev res in
  let pins = mv_pins pins [] in
  let create_lst place_ary pin =(*创建关于pin的r,another_pos*)
    let rec f i r another_pos = if i<Array.length place_ary then
    if place_ary.(i)!=None then let mos = convert_to_mos(place_ary.(i)) in
      let r = if mos.s=pin then ((float_of_int i)-.0.5)::r else r
      and another_pos = if mos.s!=pin && List.mem mos.s pins then ((float_of_int i)-.0.5)::another_pos else another_pos in
      let r = if mos.g=pin then (float_of_int i)::r else r
      and another_pos = if mos.g!=pin && List.mem mos.g pins then (float_of_int i)::another_pos else another_pos in
      let r = if mos.d=pin then ((float_of_int i)+.0.5)::r else r
      and another_pos = if mos.d!=pin && List.mem mos.d pins then ((float_of_int i)+.0.5)::another_pos else another_pos in f (i+1) r another_pos
    else f (i+1) r another_pos
  else List.rev r,List.rev another_pos
  in f 0 [] [] in
  let create pin = let pl1,pl2 = create_lst pp_ary pin and nl1,nl2 = create_lst np_ary pin in
  List.sort compare (remove_duplicates (List.append pl1 nl1)),List.sort compare (remove_duplicates(List.append pl2 nl2)) in(*并且从小到大排序*)
  let distance pos lst = (*pos的距离*)
    let ary = Array.of_list lst in
    let len = Array.length ary in
    if pos<ary.(0) then ary.(0) -. pos
    else if pos>ary.(len-1) then pos -. ary.(len-1) 
    else let rec traverse i =
      if i+1<len then 
        if pos>=ary.(i) && pos<=ary.(i+1) then min (pos-.ary.(i)) (ary.(i+1)-.pos) else traverse (i+1)
      else -1.
    in traverse 0 in
  let pin_coord pin =(*寻找pin的最大距离对应的坐标*)
    let r,another_pos = create pin in
      let rec f lst max_dis res = match lst with
      |(hd::tl) -> let dis = distance hd another_pos in
      if dis>max_dis then f tl dis hd else f tl max_dis res
      |[] -> res
  in f r min_float 0. in
  let pin_coords = Array.of_list(List.sort compare (List.map pin_coord pins)) in
  let len = Array.length pin_coords in
  if len<2 then 1.
  else let left_spacing = pin_coords.(0) +. 0.5 and right_spacing = width -. 0.5 -. pin_coords.(len-1) in
  let pin_spacing = if left_spacing>1. then [left_spacing/.width] else [] in
  let pin_spacing = if right_spacing>1. then (right_spacing/.width)::pin_spacing else pin_spacing in
  let rec func i lst =
    if i+1<len then func (i+1) (((pin_coords.(i+1) -. pin_coords.(i))/.width)::lst)
    else lst
in standard_deviation (func 0 pin_spacing)(*计算pin_spacing中的引脚间距标准差作为pin_access*)

let evaluator (w_ref:int) (nets:nets) (pins:int list) (pmos_ary:mos array) (nmos_ary:mos array) (pmos_place:mos option array) (nmos_place:mos option array) = (*目标函数,当前仅以布局宽度作为优化指标*)
  let w_ref = float_of_int w_ref in
  let pmos_lst = Array.to_list pmos_ary and nmos_lst = Array.to_list nmos_ary in
  let x_lst = List.append (List.map (function mos -> mos.x) pmos_lst) (List.map (function mos -> mos.x) nmos_lst) in
  let width = float_of_int ((List.fold_left max 0 x_lst)+1) in
  let ws = 40. *. (1. -. (width -. w_ref) /. (w_ref +. 20.)) in(*宽度分数*)
  let bbox = routing nets pmos_place nmos_place in
  let y = 20. *. (1. -. (bbox -. w_ref *. (float_of_int (List.length pins-1))) /. 60.) in
  let bs = min 20. y in(*布线分数*)
  let ss = 10. -. float_of_int (calc_symmetric pmos_place nmos_place) in(*对称性分数*)
  let pin_access = calc_pin_access nets pins pmos_place nmos_place in(*计算引脚密度分数*)
  let ps = 10. *. (1. -. pin_access) in
(ws +. bs +. ss +. ps),width,bbox,pin_access,ss

let pre_MOS (place_ary:mos option array) (pos:int) = (*获取pos位置前面MOS集合*)
  let rec f (pre_MOS:mos option list) (i:int) = 
    if i>=0 then
      if place_ary.(i)!=None then f (place_ary.(i)::pre_MOS) (i-1)
      else Array.of_list pre_MOS
    else Array.of_list pre_MOS
in f [] (pos-1)

let beh_MOS (place_ary:mos option array) (pos:int) = (*获取pos位置后面MOS集合*)
  let rec f (beh_MOS:mos option list) (i:int) = 
    if i<Array.length place_ary then
      if place_ary.(i)!=None then f (place_ary.(i)::beh_MOS) (i+1)
      else Array.of_list (List.rev beh_MOS)
    else Array.of_list (List.rev beh_MOS)
in f [] (pos+1)

let select_some ary =
  let rec f i lst =
    if i<Array.length ary then
      if ary.(i)!=None then f (i+1) ((convert_to_mos(ary.(i)))::lst)
      else f (i+1) lst
    else List.rev lst
in f 0 []

(*检查栅极共用的mos集合能否翻转*)
let flip_preMOS' (mos_set':mos option array) (bound_mos:mos option) (mv_mos':mos option) =(*mos_set是之前的mos集合,mos_set'是与mos_set栅极共用的mos集合,bound_mos是mos_set'左边的mos,mv_mos'是与移动的mos栅极共用的mos*)
  let fMOS' = if (Array.length mos_set')>0 then mos_set'.(0) else None and lMOS' = if (Array.length mos_set')>0 then mos_set'.(Array.length mos_set'-1) else None in(*只有在MOS为空时才可能为None*)
  if Array.length mos_set'=0 || Array.length mos_set'=1 then 1(*mos_set长为0/1时不需考虑栅极共用*)
  else if(bound_mos=None || lMOS'=None || (convert_to_mos(lMOS')).d=(convert_to_mos(bound_mos)).d) &&
  (fMOS'=None || mv_mos'=None || (convert_to_mos(fMOS')).s=(convert_to_mos(mv_mos')).s) then 0
  else -1

let flip_behMOS' (mos_set':mos option array) (bound_mos:mos option) (mv_mos':mos option) =
  let fMOS' = if (Array.length mos_set')>0 then mos_set'.(0) else None and lMOS' = if (Array.length mos_set')>0 then mos_set'.(Array.length mos_set'-1) else None in(*只有在MOS为空时才可能为None*)
  if Array.length mos_set'=0 || Array.length mos_set'=1 then 1(*mos_set长为0/1时不需考虑栅极共用*)
  else if(mv_mos'=None || lMOS'=None || (convert_to_mos(lMOS')).d=(convert_to_mos(mv_mos')).d) &&
  (fMOS'=None || bound_mos=None || (convert_to_mos(fMOS')).s=(convert_to_mos(bound_mos)).s) then 0
  else -1

(*注意到record中每个元素array之间是没有交集的*)
let record_flip (record:mos option array list) (mos_ary:mos array) =(*返回需要翻转的mos_lst,需要区分类型*)
  let lsts = List.map Array.to_list record in
  let lst = List.fold_left List.append [] lsts in
  let some_lst = select_some (Array.of_list lst) in
List.map (function mos -> mos.id) some_lst(*需要翻转的mos id*)

let set_mos_flip (mos_ary:mos array) (place_ary:mos option array) (mos_set_flip:mos option array list) = (*更新翻转*)
  let flip = record_flip mos_set_flip mos_ary in
  let rec f lst = match lst with
  |(hd::tl) -> let mos=mos_ary.(hd) in 
  let s=mos.s and d=mos.d in
  Array.set mos_ary hd {mos with s=d;d=s};
  f tl
  |[] ->
  for i=0 to Array.length place_ary-1 do(*用flip更新物理布局*)
    let mos = convert_to_mos(place_ary.(i)) in
    if place_ary.(i)!=None && List.mem mos.id flip then
    let s=mos.s and d=mos.d in
    Array.set place_ary i (Some{mos with s=d;d=s});
  done
in f flip

let set_subary_flip (place_ary) (pos:int) (len:int) = (*从pos开始长为len的sub_ary reverse*)
  if pos>=0 && pos<Array.length place_ary && len>0 then 
  Array.append (Array.append (Array.sub place_ary 0 pos) (Array.of_list(List.rev(Array.to_list( Array.sub place_ary pos len))))) (Array.sub place_ary (pos+len) (Array.length place_ary-len-pos)) 
  else place_ary

(*只是翻转移动mos管优化力度不够，必须考虑翻转前面的mos集合后面的mos集*)
let is_legal (pary:mos array) (nary:mos array) (pplace_ary:mos option array) (nplace_ary:mos option array) (mv_occ:int) (mv_emp:int) = (*合法,翻转preMOS,翻转mv_mos,翻转behMOS*)
  let pmos_ary = Array.copy pary and nmos_ary = Array.copy nary in
  let pp_ary = Array.copy pplace_ary and np_ary = Array.copy nplace_ary in
  let mv_mos = pp_ary.(mv_occ) in
  Array.set pp_ary mv_occ None;(*如果是false的话，需要将mv_mos置回*)
  let preMOS = pre_MOS pp_ary mv_emp and behMOS = beh_MOS pp_ary mv_emp in(*pos之前的mos集合，pos之后的mos集合*)
  let preMOS' = Array.sub np_ary (mv_emp-Array.length preMOS) (Array.length preMOS)
  and behMOS' = Array.sub np_ary (mv_emp+1) (Array.length behMOS) in(*与preMOS匹配的*)
  let fpreMOS = if (Array.length preMOS)>0 then preMOS.(0) else None and lpreMOS = if (Array.length preMOS)>0 then preMOS.(Array.length preMOS-1) else None in(*只有在MOS为空时才可能为None*)
  let fbehMOS = if (Array.length behMOS)>0 then behMOS.(0) else None and lbehMOS = if (Array.length behMOS)>0 then behMOS.(Array.length behMOS-1) else None in
  let left_preMOS' = let idx = mv_emp-(Array.length preMOS)-1 in if idx<0 then None else np_ary.(idx) in(*匹配的preMOS'左边第一个mos*)
  let right_behMOS' = let idx = mv_emp+(Array.length behMOS)+1 in if idx>=Array.length np_ary then None else np_ary.(idx) in
    (* 0 0 0 *)
    if ((lpreMOS=None || (convert_to_mos(lpreMOS)).d=(convert_to_mos(mv_mos)).s) && 
    (fbehMOS=None || (convert_to_mos(mv_mos)).d=(convert_to_mos(fbehMOS)).s) &&(*左右可以源漏共用*)
    ((np_ary.(mv_emp)=None || (convert_to_mos(mv_mos)).g=(convert_to_mos(np_ary.(mv_emp))).g)))(*栅极可以共用*) then begin
      Array.set pp_ary mv_emp mv_mos;
    true,pmos_ary,nmos_ary,pp_ary,np_ary end

    (* 0 0 1 *)
    else if (lpreMOS=None || (convert_to_mos(lpreMOS)).d=(convert_to_mos(mv_mos)).s) && 
    (lbehMOS=None || (convert_to_mos(mv_mos)).d=(convert_to_mos(lbehMOS)).d) &&
    ((np_ary.(mv_emp)=None || (convert_to_mos(mv_mos)).g=(convert_to_mos(np_ary.(mv_emp))).g)) then
    let res = flip_behMOS' behMOS' right_behMOS' np_ary.(mv_emp) in
    if res=1 then begin(*翻转behMOS*)
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [behMOS];
      let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) in
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    else if res=0 then begin(*翻转behMOS,behMOS'*)
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [behMOS];
      set_mos_flip nmos_ary np_ary [behMOS'];
      let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) and np_ary = set_subary_flip np_ary (mv_emp+1) (Array.length behMOS') in
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    else begin
      Array.set pp_ary mv_occ mv_mos;(*置回*)
    false,pmos_ary,nmos_ary,pp_ary,np_ary end

    (* 0 1 0 *)
    else if (lpreMOS=None || (convert_to_mos(lpreMOS)).d=(convert_to_mos(mv_mos)).d) &&
    (fbehMOS=None || (convert_to_mos(mv_mos)).s=(convert_to_mos(fbehMOS)).s) &&(*左右可以源漏共用*)
    ((np_ary.(mv_emp)=None || (convert_to_mos(mv_mos)).g=(convert_to_mos(np_ary.(mv_emp))).g)) then begin
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [Array.of_list [mv_mos]];
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    
    (* 0 1 1 *)
    else if (lpreMOS=None || (convert_to_mos(lpreMOS)).d=(convert_to_mos(mv_mos)).d) && 
    (lbehMOS=None || (convert_to_mos(mv_mos)).s=(convert_to_mos(lbehMOS)).d) &&
    ((np_ary.(mv_emp)=None || (convert_to_mos(mv_mos)).g=(convert_to_mos(np_ary.(mv_emp))).g)) then
    let res = flip_behMOS' behMOS' right_behMOS' np_ary.(mv_emp) in
    if res=1 then begin(*翻转behMOS,mv_mos*)
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [behMOS;Array.of_list [mv_mos]];
      let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) in
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    else if res=0 then begin(*翻转behMOS,mv_mos,behMOS'*)
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [behMOS;Array.of_list [mv_mos]];
      set_mos_flip nmos_ary np_ary [behMOS'];
      let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) and np_ary = set_subary_flip np_ary (mv_emp+1) (Array.length behMOS') in
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    else begin
      Array.set pp_ary mv_occ mv_mos;(*置回*)
    false,pmos_ary,nmos_ary,pp_ary,np_ary end

    (* 1 0 0 *)
    else if (fpreMOS=None || (convert_to_mos(fpreMOS)).s=(convert_to_mos(mv_mos)).s) && 
    (fbehMOS=None || (convert_to_mos(mv_mos)).d=(convert_to_mos(fbehMOS)).s) &&
    ((np_ary.(mv_emp)=None || (convert_to_mos(mv_mos)).g=(convert_to_mos(np_ary.(mv_emp))).g)) then
    let res = flip_preMOS' preMOS' left_preMOS' np_ary.(mv_emp) in
    if res=1 then begin(*翻转preMOS*)
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [preMOS];
      let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) in
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    else if res=0 then begin(*翻转preMOS,preMOS'*)
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [preMOS];
      set_mos_flip nmos_ary np_ary [preMOS'];
      let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) and np_ary = set_subary_flip np_ary (mv_emp-Array.length preMOS') (Array.length preMOS') in
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    else begin
      Array.set pp_ary mv_occ mv_mos;(*置回*)
    false,pmos_ary,nmos_ary,pp_ary,np_ary end

    (* 1 0 1 *)
    else if (fpreMOS=None || (convert_to_mos(fpreMOS)).s=(convert_to_mos(mv_mos)).s) && 
    (lbehMOS=None || (convert_to_mos(mv_mos)).d=(convert_to_mos(lbehMOS)).d) &&
    (np_ary.(mv_emp)=None || (convert_to_mos(mv_mos)).g=(convert_to_mos(np_ary.(mv_emp))).g) then
    let res1 = flip_preMOS' preMOS' left_preMOS' np_ary.(mv_emp) and res2 = flip_behMOS' behMOS' right_behMOS' np_ary.(mv_emp) in
    if res1=1 then
      if res2=1 then begin(*翻转preMOS,behMOS*)
        Array.set pp_ary mv_emp mv_mos;
        set_mos_flip pmos_ary pp_ary [preMOS;behMOS];
        let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) in
        let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) in
      true,pmos_ary,nmos_ary,pp_ary,np_ary end
      else if res2=0 then begin(*翻转preMOS,behMOS,behMOS'*)
        Array.set pp_ary mv_emp mv_mos;
        set_mos_flip pmos_ary pp_ary [preMOS;behMOS];
        set_mos_flip nmos_ary np_ary [behMOS'];
        let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) and np_ary = set_subary_flip np_ary (mv_emp+1) (Array.length behMOS') in
        let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) in
      true,pmos_ary,nmos_ary,pp_ary,np_ary end
      else begin
        Array.set pp_ary mv_occ mv_mos;(*置回*)
      false,pmos_ary,nmos_ary,pp_ary,np_ary end
    else if res1=0 then(*翻转preMOS,behMOS,preMOS'*)
      if res2=1 then begin
        Array.set pp_ary mv_emp mv_mos;
        set_mos_flip pmos_ary pp_ary [preMOS;behMOS];
        set_mos_flip nmos_ary np_ary [preMOS'];
        let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) and np_ary = set_subary_flip np_ary (mv_emp-Array.length preMOS') (Array.length preMOS') in
        let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) in
      true,pmos_ary,nmos_ary,pp_ary,np_ary end
      else if res2=0 then begin(*翻转preMOS,behMOS,preMOS'和behMOS'*)
        Array.set pp_ary mv_emp mv_mos;
        set_mos_flip pmos_ary pp_ary [preMOS;behMOS];
        set_mos_flip nmos_ary np_ary [preMOS';behMOS'];
        let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) and np_ary = set_subary_flip np_ary (mv_emp-Array.length preMOS') (Array.length preMOS') in
        let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) and np_ary = set_subary_flip np_ary (mv_emp+1) (Array.length behMOS') in
      true,pmos_ary,nmos_ary,pp_ary,np_ary end
      else begin
        Array.set pp_ary mv_occ mv_mos;(*置回*)
      false,pmos_ary,nmos_ary,pp_ary,np_ary end
    else begin
      Array.set pp_ary mv_occ mv_mos;(*置回*)
    false,pmos_ary,nmos_ary,pp_ary,np_ary end

    (* 1 1 0 *)
    else if (fpreMOS=None || (convert_to_mos(fpreMOS)).s=(convert_to_mos(mv_mos)).d) && 
    (fbehMOS=None || (convert_to_mos(mv_mos)).s=(convert_to_mos(fbehMOS)).s) &&
    ((np_ary.(mv_emp)=None || (convert_to_mos(mv_mos)).g=(convert_to_mos(np_ary.(mv_emp))).g)) then
    let res = flip_preMOS' preMOS' left_preMOS' np_ary.(mv_emp) in
    if res=1 then begin(*翻转preMOS,mv_mos*)
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [preMOS;Array.of_list [mv_mos]];
      let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) in
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    else if res=0 then begin(*翻转preMOS,mv_mos,preMOS'*)
      Array.set pp_ary mv_emp mv_mos;
      set_mos_flip pmos_ary pp_ary [preMOS;Array.of_list [mv_mos]];
      set_mos_flip nmos_ary np_ary [preMOS'];
      let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) and np_ary = set_subary_flip np_ary (mv_emp-Array.length preMOS') (Array.length preMOS') in
    true,pmos_ary,nmos_ary,pp_ary,np_ary end
    else begin
      Array.set pp_ary mv_occ mv_mos;(*置回*)
    false,pmos_ary,nmos_ary,pp_ary,np_ary end

    (* 1 1 1 *)
    else if (fpreMOS=None || (convert_to_mos(fpreMOS)).s=(convert_to_mos(mv_mos)).d) && 
    (lbehMOS=None || (convert_to_mos(mv_mos)).s=(convert_to_mos(lbehMOS)).d) &&
    ((np_ary.(mv_emp)=None || (convert_to_mos(mv_mos)).g=(convert_to_mos(np_ary.(mv_emp))).g)) then
    let res1 = flip_preMOS' preMOS' left_preMOS' np_ary.(mv_emp) and res2 = flip_behMOS' behMOS' right_behMOS' np_ary.(mv_emp) in
    if res1=1 then
      if res2=1 then begin(*翻转preMOS,mv_mos,behMOS*)
        Array.set pp_ary mv_emp mv_mos;
        set_mos_flip pmos_ary pp_ary [preMOS;Array.of_list [mv_mos];behMOS];
        let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) in
        let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) in
      true,pmos_ary,nmos_ary,pp_ary,np_ary end
      else if res2=0 then begin(*翻转preMOS,mv_mos,behMOS,behMOS'*)
        Array.set pp_ary mv_emp mv_mos;
        set_mos_flip pmos_ary pp_ary [preMOS;Array.of_list [mv_mos];behMOS];
        set_mos_flip nmos_ary np_ary [behMOS'];
        let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) and np_ary = set_subary_flip np_ary (mv_emp+1) (Array.length behMOS') in
        let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) in
      true,pmos_ary,nmos_ary,pp_ary,np_ary end
      else begin
        Array.set pp_ary mv_occ mv_mos;(*置回*)
      false,pmos_ary,nmos_ary,pp_ary,np_ary end
    else if res1=0 then
      if res2=1 then begin(*翻转preMOS,mv_mos,behMOS,preMOS'*)
        Array.set pp_ary mv_emp mv_mos;
        set_mos_flip pmos_ary pp_ary [preMOS;Array.of_list [mv_mos];behMOS];
        set_mos_flip nmos_ary np_ary [preMOS'];
        let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) and np_ary = set_subary_flip np_ary (mv_emp-Array.length preMOS') (Array.length preMOS') in
        let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) in
      true,pmos_ary,nmos_ary,pp_ary,np_ary end
      else if res2=0 then begin(*翻转preMOS,mv_mos,behMOS,preMOS'和behMOS'*)
        Array.set pp_ary mv_emp mv_mos;
        set_mos_flip pmos_ary pp_ary [preMOS;Array.of_list [mv_mos];behMOS];
        set_mos_flip nmos_ary np_ary [preMOS';behMOS'];
        let pp_ary = set_subary_flip pp_ary (mv_emp-Array.length preMOS) (Array.length preMOS) and np_ary = set_subary_flip np_ary (mv_emp-Array.length preMOS') (Array.length preMOS') in
        let pp_ary = set_subary_flip pp_ary (mv_emp+1) (Array.length behMOS) and np_ary = set_subary_flip np_ary (mv_emp+1) (Array.length behMOS') in
      true,pmos_ary,nmos_ary,pp_ary,np_ary end
      else begin
        Array.set pp_ary mv_occ mv_mos;(*置回*)
      false,pmos_ary,nmos_ary,pp_ary,np_ary end
    else begin
      Array.set pp_ary mv_occ mv_mos;(*置回*)
    false,pmos_ary,nmos_ary,pp_ary,np_ary end

  else begin
    Array.set pp_ary mv_occ mv_mos;(*置回*)
  false,pmos_ary,nmos_ary,pp_ary,np_ary end

let compute ary =(*计算some none出现位置*)
  let rec f i some_lst none_lst = if i<Array.length ary then
    if ary.(i)=None then f (i+1) some_lst (i::none_lst) else f (i+1) (i::some_lst) none_lst
    else Array.of_list (List.rev some_lst),Array.of_list (List.rev none_lst)
in f 0 [] []

let print_mos (mos:mos option) =
  let mos = convert_to_mos(mos) in
  printf "(%i,%i,%i)\t" mos.s mos.g mos.d

let print_place_ary (place:mos option array) =
  for i=0 to Array.length place-1 do
    if place.(i)=None then print_string "None\t"
    else print_mos place.(i);
  done

let simulated_annealing (w_ref:int) (nets:nets) (pins:int list) (pmos_ary : mos array) (nmos_ary : mos array) (pp_ary : mos option array) (np_ary : mos option array) (t0 : float) (tt : float) (decrease : float) (times : int)=(*t0初始温度，tt终止温度，increase每次温度的变化量(0,1)，times在每个温度下训练次数*)
  (* let mos_num = Array.length pmos_ary + Array.length nmos_ary in *)
  let rec for_each_turn pmos_ary nmos_ary pp_ary np_ary t tp turn:mos array * mos array * mos option array * mos option array =(*turn表示当前进行的轮次,tp表示当前轮次是对nmos:false pmos:true进行操作*)
    if turn<times then
    begin
      let occupy,empty = compute (if tp then pp_ary else np_ary) in
      (*if Array.length empty=0 then pmos_ary,nmos_ary,pp_ary,np_ary else*)(*这里还需要修改，保证empty的长度不会为0*)
      let i = Random.int (Array.length occupy) and j = Random.int (Array.length empty) in
      let legal,pary,nary,pplace,nplace = if tp=true then is_legal pmos_ary nmos_ary pp_ary np_ary occupy.(i) empty.(j) else is_legal nmos_ary pmos_ary np_ary pp_ary occupy.(i) empty.(j) in
      let new_pary = if tp=true then pary else nary and new_nary = if tp=true then nary else pary in
      let new_pp = if tp=true then pplace else nplace and new_np = if tp=true then nplace else pplace in
      if legal && (check_notch new_pp) && (check_notch new_np) then(*布局合法 且 不出现notch*)
      begin
        let new_pp,new_np = update_place_ary new_pp new_np in(*整体前移,过程中也应该保持不出现notch*)
        update_mos_ary new_pary (Array.to_list new_pp);(*用物理布局更新mos_ary*)
        update_mos_ary new_nary (Array.to_list new_np);
        let s0,_,_,_,_ = evaluator w_ref nets pins pmos_ary nmos_ary pp_ary np_ary and s1,_,_,_,_ = evaluator w_ref nets pins new_pary new_nary new_pp new_np in
        let delta = s1 -. s0 in
        if delta>0. then begin
          (*在new_place后添加两列None，加入接收准则*)
          for_each_turn new_pary new_nary (Array.append new_pp [|None;None|]) (Array.append new_np [|None;None|]) t (Bool.not tp) (turn+1)(*接受新解*)
        end
        else for_each_turn pmos_ary nmos_ary pp_ary np_ary t tp (turn+1)
      end
      else for_each_turn pmos_ary nmos_ary pp_ary np_ary t tp (turn+1)
    end
    else pmos_ary,nmos_ary,pp_ary,np_ary in
  let rec for_each_temp pmos_ary nmos_ary pp_ary np_ary t =
    if t>tt then let new_pary,new_nary,new_pp,new_np = for_each_turn pmos_ary nmos_ary pp_ary np_ary t true 0 in
    for_each_temp new_pary new_nary new_pp new_np (t *. decrease)
    else pmos_ary,nmos_ary,pp_ary,np_ary
in for_each_temp pmos_ary nmos_ary (Array.append pp_ary [|None;None|]) (Array.append np_ary [|None;None|]) t0

(* let write_to_record (cells:string) (cell:string) = (*生成布局，将布局评分记录到record.txt文件中*)
  let start_time = Sys.time () in(*开始*)
  let s = read_cell cells cell in
  let (pins,nets,nmos,pmos,filename,w_ref) = init s in
  let mos_num = List.length pmos + List.length nmos in
  let nets:nets = Array.of_list nets in
  let pmos_ary,nmos_ary,pp_ary,np_ary = init_layout pmos nmos in(*mos_place为mos管的物理布局*)
  let filename = filename ^ ".json" in
  let new_pary,new_nary,new_pp,new_np = simulated_annealing w_ref nets pins pmos_ary nmos_ary pp_ary np_ary (float_of_int mos_num *. 20000.) 0.1 0.88 (mos_num * 2000) in
  let content = placement_str new_pary new_nary nets in
  write_to_file filename content;
  let end_time = Sys.time () in(*结束*)
  let elapsed_time = end_time -. start_time in
  print_place_ary new_pp;printf"\n";
  print_place_ary new_np;printf"\n";
  let _,width,bbox,pin_access,symmetric = evaluator w_ref nets pins new_pary new_nary new_pp new_np in
  let drc = if check_notch new_pp && check_notch new_np then 10 else 0 in 
  let line = "Cell:" ^ cell ^ "\twidth:"^ string_of_int (int_of_float width) ^ "\tbbox:" ^ string_of_float bbox 
  ^ "\tpin_access:" ^ string_of_float pin_access ^ "\tsymmetric:" ^ string_of_int (int_of_float symmetric) 
  ^ "\tdrc:" ^ string_of_int drc ^ "\truntime:" ^ string_of_float elapsed_time ^ "s\n" in
write_to_file' "record.txt" line *)

(* let find_cell_name (name_lst:string list) (cell:string) = 
  let rec f lst res = match lst with
  |(hd::tl) -> if hd=cell then res else f tl (res+1)
  |[] -> -1
in f name_lst 0

let name_lst = read_cells_name "cells.spi"
*)
let test (cells:string) (cell:string) =
  let s = read_cell cells cell in
  let (pins,nets,nmos,pmos,filename,w_ref) = init s in
  let mos_num = List.length pmos + List.length nmos in
  let nets:nets = Array.of_list nets in
  let pmos_ary,nmos_ary,pp_ary,np_ary = init_layout pmos nmos in(*mos_place为mos管的物理布局*)
  let filename = filename ^ ".json" in
  let new_pary,new_nary,new_pp,new_np = simulated_annealing w_ref nets pins pmos_ary nmos_ary pp_ary np_ary (float_of_int mos_num *. 20000.) 0.1 0.88 (mos_num * 2000) in
  let content = placement_str new_pary new_nary nets in
write_to_file filename content

let main () =
  let arguments = Array.to_list Sys.argv in
  match arguments with
  | [_; cells; cell] -> test cells cell
  | _ ->
    print_endline "Usage: eda.exe <netlist> <cell_name>"

let () = main ()