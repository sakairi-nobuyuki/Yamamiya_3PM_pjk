# 設計検討書

## 1. 装置本体

### 1.1. 一時保存容器



### 1.2. ベルトコンベヤ
#### 1.2.1. 基本設計

- 梅1ケを運搬するものとして、コンベヤの幅は平均的な梅の直径30mmに対してマージンを持たせて50mmにする。
- 梅の密度を1000 kg/m3として、形状が球と仮定すると、その重量は65 g。
- コンベヤは、一時保存容器の底部から一時保存容器の上端部まで梅を運搬できるようにする。コンベヤを垂直配置すると梅がうまく運べないので、斜め配置にする。角度は45°にしておく。
- 上記のように、コンベヤは一時保存容器の底部から梅を掬いあげていくので、コンベヤベルトは桟付きのものにする。桟の幅はコンベヤ幅と同じように50mmとする。
- 容器の高さを1mとするため、コンベヤの運搬長は高さ1mに対応するものとする。
- コンベヤの運搬能力は、コンベヤの底部から上部まで物を運ぶのに必要な力として、仮に一時保存容器の底部にかかる静水圧を仮定する。コンベヤの幅<i>d</i>、コンベヤの長さ<i>l</i>、一時保存容器の高さ<i>h</i>、および、梅の密度を<i>&rho;</i>と仮定して、荷重<i>W</i>は、$W = \rho d l h g$とする。今、<i>d</i> = 50mm、<i>l</i> = 1000mm、<i>h</i> = 1000mm、梅の密度は水の密度と同じと仮定して<i>&rho;</i> = 1000 kg/m3、<i>g</i> = 9.8 m/s2なので、<i>W</i> = 490Nが最大でかかる荷重と仮定する。



#### 1.2.2. 外形寸法



### 1. 3. 梅のフィーダ

#### 1. 3. 1. 軸受け
- 軸受の種類:
    - 下記のように低回転で荷重が小さいので、オイレスを使う。
- 形状:
    - 内径: 駆動軸の形状からφ8+0.060/+0.120。
    - 外形: φ10+0.045/+0.157
- 型式: オイレス 80F-0803
- ラジアル荷重荷重: 
    - 常用荷重: 軸受けピンの寸法を基準にして、梅の荷重が大きく見て、梅3ケ分の荷重だとすると、3 X 65 g X 9.8 N/s2 = 1.9 N。
    - 最大衝撃荷重: 常用荷重の10倍と仮定して9.6N。
- 回転数:
    - 許容最低回転数: 梅の選別が1sと仮定して、1sに一回梅を検知部に供給できればいいので、60min-1。
    - 常用回転数: 最低許容回転数の10倍の600min-1を想定しておく。軸受けサイズからあきらかに最大回転数より低い。
    - 最大回転数: 軸受けの許容最大回転数。
- スラスト荷重: 考えない。
- 座金、スペーサ: 本体の部材保護のために平座金をかまして面圧を下げる。当たり面が面積比で4倍以上になるようにする。


### 1. 3. 2. 駆動系
- 駆動軸:
    - 軸径:
        - フィーダに対して締め付け構造を持たなければならない。締め付け構造を成立させるため、ボルトの強度面からボルトサイズはM4でなければならない。
        - 汎用品で、フィーダに対して直角度を持たせる駆動軸の軸単締め付け構造はメネジ構造しかないので、フィーダとの締結面は軸端メネジ。
        - フィーダ締結面のネジサイズがM4で、駆動軸の肉厚を考えると軸径はφ8 H7。
    - 素材:
        - 軸受けはオイレスを使い、滑り軸受になるので、比較的硬度の高いS45C材以上の硬度を持つものにする。
        - 同様に、滑り軸受なので、面粗度はRa0.8が望ましいが、購入品でRa 0.8はないので、Ra 1.6とする。
        - 摺動性を確保するため、硬度が上がるような熱処理、表面処理があると望ましい。
        - 回転体なので、幾何公差の表記をよく見て決める。
    - 動バランス:
        - 回転数が低いので考慮しない。
- フィーダ締結ボルト:
    - ネジサイズ: 上記駆動軸の設計からM4。
    - 素材: 
        - そこそこの締結力を持たせる必要があるので、ISO 10.8以上材料にする。
        - 防錆のことを考えて黒メッキしてあるものがよい。
        - 焼き付きのリスクがあるのでSUS系の素材は使わない。
    - 形状:
        - サイズダウンのため六角頭がよい。
        - 鍔と軸部の垂直度を確保するため、ISOボルトがよい。
- プーリ:
    - 被駆動側常用トルク: フィーダ径がφ70で、常用荷重が1.9N なので、0.035 X 1.9 = 67.3 mNm。
    - 被駆動側最大トルク: 最大荷重が常用荷重の10倍なので、最大トルクは常用トルクの10倍の673 mNm。
    - 最大動力: 常用回転数、最大トルクで計算すると、2π NT / 60 = 2π X 600 X 0.673 / 60 =  4.23 W。
    - 最小必要動力: 最低許容回転数、常用トルクで計算すると、2π NT / 60 = 2π X 60 X 0.0673 / 60 = 0.423 W。
    - 減速比: モータの回転数と、フィーダの常用回転数から、14800 / 60 = 246
    - 最大回転数:
    - 材料:
- ベルト:
    - 最大荷重:
    - 材料: 
- モータ: ACの誘導期の単相制御が面倒なので、小型DCモータを使う。
    - モータ形式: マブチRS-540SH
    - 常用回転数: 14800 1/min-1
    - 無負荷回転数: 15800 1/min-1
    - トルク: 19.6 mNm
    - 電圧: 4.5 - 9.6 V
    - 適正電圧: 7.2 V
    - 軸径: 3.17 mm
- 試作用モータ: タミヤのミニ四駆用の減速機付きのを使う。
  - 形式: ITEM 70189-860
  - 電源: DC 3V 
  - トルク: 0.14269 Nm
  - 回転数: 9 min-1
  - 出力: 0.138 W
  - 計画電流: 46 mA
  - モータドライバ: TC78H651AFNG

## 1. 4. 梅の分別器

塩ビ製の勾配通路を梅が転がるところを、ソレノイドを使って方向を変化させて梅を分別する。

### 1. 4. 1. 方向変換装置全般

通路を斜めに跳ね上げることで方向を変化させる。

下り勾配5°に対して、きちんと方向転換できるようにする。そのため、ソレノイドのストロークは10mm以上必要。

ソレノイドの駆動にリレーが必要で、リレーの駆動にトランジスタ回路が必要。トランジスタとラズパイがつながっている。

### 1. 4. 2. アクチュエータ側

- ソレノイドの定格値: 
  - 型式: Panasonic HS SOLENOID AS45051
  - 電圧: 100V 5-60Hz
  - 電流@AC100V 5-60Hz:
    - 始動電流: 1.6 A
    - 保持電流: 0.2 A
  - 定格荷重: 
    - 吸引力: 4.9 N
    - 静荷重: 14.7 N
  - ストローク: 15mm

### 1. 4. 3. ソレノイドドライバ

- 設計方針:
  - AC100Vのソレノイドを駆動しないといけないので、DC24V駆動の機械式のリレーを使う。よく使われるMY2にする。
  - リレーはサージキラー付きのMY2-Dにする。
  - チャタリング防止にコンデンサと抵抗を入れる。

- リレーの選定:
  - 型式: OMRON MY2-D DC24
  - コイル電圧: DV24V 36.3mA 662Ω
  - 接点定格電圧: AC220V 5A

- コンデンサ  
　- 選定基準: 接点電流1Aに対し0.5～1（μF）、DCの場合は不要。
    https://ac-blog.panasonic.co.jp/%E3%83%AA%E3%83%AC-%E3%82%B3%E3%82%A4%E3%83%AB%E3%81%AE%E3%82%B5-%E3%82%B8%E4%BF%9D%E8%AD%B7%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
  

- 抵抗:
  - 選定基準: 接点電圧1Vに対し0.5～1（Ω）、ACの場合は不要。
  https://ac-blog.panasonic.co.jp/%E3%83%AA%E3%83%AC-%E3%82%B3%E3%82%A4%E3%83%AB%E3%81%AE%E3%82%B5-%E3%82%B8%E4%BF%9D%E8%AD%B7%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6

### 1. 4. 4. リレードライバ

- 設計方針:
  - 自分でアナログ回路を組むと大変なので、既存のICを使う。

- ソレノイドドライバ:
  - 型式: TB67S112PG
  - 電源電圧: DC 24V
  - 出力側: DC 24V
  - 入力電圧: DC 2.0-5.5V
  - メーカー: 東芝

- ダイオード: 
  - Select the zener diode that meets the condition as follows; VM + VFN + VZ < output rating of 50V

- コンデンサ: 
  - 47μF



### 1. 4. 4. 電源装置

- 電源の選定: 
  - DC 24Vの負荷:
    - リレー: 36.3 mA
    - DC 3V: 1 A
    - 合計: 2 A
  - DC 3Vの負荷:
    - モータ: 46 mA X 3 = 150 mA
    - RasPi: 600mA X 5 = 3 A
    - 合計: 4A