# 第 05 講｜AI 怎麼記得前面說過什麼？RNN、Attention 與 Transformer

> 課程：生成式 AI 文字與圖像（115 學年度上學期）
> 資料來源：投影片《05 AI 怎麼記得前面說過什麼？》（86 頁）＋ 上課錄影（YouTube <https://www.youtube.com/watch?v=vmQzpZvVqaI>，約 2 小時 43 分）
> 逐字稿：[影片/05_逐字稿.md](影片/05_逐字稿.md)
> 說明：本講義依投影片架構整理，並補上老師上課的口頭解說（標示為 💬 老師補充）。

---

## 本講重點總覽

| 單元 | 主題 | 一句話重點 |
|---|---|---|
| 01 | RNN 的原理 | RNN 神經元就是一般神經元，只是**多吃了上一次自己的輸出（hidden states）** |
| 02 | 線性代數 101 | 矩陣**先列後行**；乘法的核心是**內積**；向量乘矩陣＝對矩陣的**列向量做線性組合** |
| 03 | QKV 注意力機制 | $\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$：用內積算相關強度，再對 value 做加權平均 |
| 04 | Position Encoding | Transformer 一次平行讀入，所以要用 sin／cos「時鐘」把**位置資訊**加進 embedding |
| 05 | 讓 Transformer 更穩 | **Residual 連結**＋**Layer Normalization**；新模型改用 RMSNorm，Meta 甚至提出 DyT（💬 上課略講） |
| 作業 | Vibe Coding 互動網頁 | 和 AI 討論出**規格書** → 產生 `index.html` → 用 **GitHub Pages** 發佈 |
| 附錄 | 助教時間 | 兩位同學閃電秀（skill.md 當 System Prompt、Usagi Agent）＋作業四範例分享 |

**最重要的觀念**：Transformer 沒有什麼神奇的地方。它就是讓一串字**平行地**互相比較（attention），決定每個位置的「記憶」（hidden state）應該長什麼樣子。真正訓練的，只是 $W^Q, W^K, W^V$ 這幾個矩陣。

---

## 💬 課前說明（影片約 0:00 起）

- 今天大概是**最後一次介紹比較技術的東西**。作業跟之前一樣，沒有弄到那麼技術，不用太擔心。
- 不要說「技術」，聽起來很嚴肅，應該說是「原理」。原理多理解一下，你比較會知道發生了什麼事。
- 數學覺得太困難沒關係，**反正一輩子總要有一次**，聽一次大家在吹的 Transformer 到底發生了什麼事。

---

## 01. RNN 的原理（投影片 P2–P8，影片約 0:01 起）

💬 老師補充：今天要討論的是**有記憶的神經網路**。上次很快地提過，RNN 是有記憶神經網路的代表，特別是在「古早古早以前」。

### 1.1 遞歸層原理（P3–P5）

- 用「兩個輸入、兩個 RNN 神經元」的遞歸層來說明，專注在**第一個 RNN 神經元**。
- 看起來好像和以前的全連結層一樣……
- **不一樣的是：上次 RNN 層的輸出還會再傳回來**（$h_{t-1}^1, h_{t-1}^2$），然後才決定這次的輸出。

$$
h_t^1=\sigma\left(w_{11}^x x_t^1+w_{21}^x x_t^2+w_{11}^h h_{t-1}^1+w_{21}^h h_{t-1}^2+b_1\right)
$$

💬 老師補充：
- 這一層的輸出就叫 **hidden state**，也就是我們說的「**記憶**」。記憶其實**只是數字**：兩個神經元在 $t-1$ 時間點吐出來的兩個數字。
- 因為這一層有兩個神經元，記憶就是兩個數字；它們再傳回來，**就好像這個神經元接到四個數字一樣**。
- 算法跟以前的全連結**一模一樣**：加權和、加上偏值，經過 activation function 送出去。公式看起來可怕，只是因為要標出是哪一個時間點。

### 1.2 認真看，RNN 神經元就是一般的神經元（P6–P8）

- 第 $k$ 號 RNN 神經元的輸入分成兩組：「正常」輸入 $\mathbf{x}_t$，和**上次的 hidden states** $\mathbf{h}_{t-1}$。
- 寫成矩陣：

$$
(W^X)^T=\begin{bmatrix}w_{11}^x & w_{21}^x\\ w_{12}^x & w_{22}^x\end{bmatrix},\quad
\mathbf{x}_t=\begin{bmatrix}x_t^1\\ x_t^2\end{bmatrix},\quad
(W^X)^T\mathbf{x}_t=\begin{bmatrix}w_{11}^x x_t^1+w_{21}^x x_t^2\\ w_{12}^x x_t^1+w_{22}^x x_t^2\end{bmatrix}
$$

- 一次矩陣乘法就**同時算了兩個神經元**；hidden states 那邊 $(W^H)^T\mathbf{h}_{t-1}$ 基本上也一樣。

---

## 02. 線性代數 101（投影片 P9–P14，影片約 0:04 起，老師先在黑板推導）

💬 老師補充：**推薦大家修線性代數**
- 如果你還是大學部的同學，**只想學一門數學，非常推薦線性代數**；可以學兩門的話，就是**線性代數＋微積分**（包含高微）。這兩大系列弄清楚，幾乎所有數學都不會再困擾你。
- 要修的話，非常推薦到**應用數學系或數學系**去修。

### 2.1 矩陣最大的關鍵：先列後行（P10）

- $m\times n$ 矩陣：$m$ **列**（橫的）、$n$ **行**（直的）；$a_{ij}$ 是第 $i$ 列、第 $j$ 行的元素。

$$
A=\begin{bmatrix}a_{11}&a_{12}&\cdots&a_{1n}\\ a_{21}&a_{22}&\cdots&a_{2n}\\ \vdots&&&\vdots\\ a_{m1}&a_{m2}&\cdots&a_{mn}\end{bmatrix}=[a_{ij}]_{m\times n}
$$

💬 老師補充：
- 口號就是「**先列後行**」：**列是 Row（橫的）、行是 Column（直的）**。記得這件事，線性代數幾乎就沒什麼問題。
- ⚠️ **台灣的「列／行」和中國大陸、日本是相反的**：人家的「行」是我們的「列」。如果你原本學的是那套系統，就記得相反就好。
- 現場問答：第一列第一個元素是 $a_{11}$，下一個是 $a_{12}$ 還是 $a_{21}$？先列後行，所以是 **$a_{12}$**。
- 習慣上**矩陣用大寫**（$A$），**裡面的元素用小寫**（$a_{ij}$）。
- 剛開始不習慣抽象寫法，就想成 $2\times3$ 或 $3\times2$ 的矩陣，**不要讓 $m$ 和 $n$ 一樣**，比較不會搞混。

### 2.2 乘法也是先列後行（P11）

- 若 $C=AB$（$A$ 是 $m\times n$、$B$ 是 $n\times p$，$C$ 就是 $m\times p$），則 $c_{ij}$ 是 **A 的第 $i$ 列**和 **B 的第 $j$ 行**的內積：

$$
c_{ij}=a_{i1}b_{1j}+a_{i2}b_{2j}+\cdots+a_{in}b_{nj}
$$

💬 老師補充：以後看到任何矩陣相乘，都要很快有這個反應——**前面出列、後面出行**。規則忘了就問 AI，讓它給你一個例子，你乘兩次就知道了。

### 2.3 內積（dot product）是核心（P12）

$$
\mathbf{u}=[a_1,\dots,a_p],\ \mathbf{v}=[b_1,\dots,b_p]\ \Rightarrow\ \langle \mathbf{u},\mathbf{v}\rangle=\mathbf{u}\mathbf{v}^T=a_1b_1+a_2b_2+\cdots+a_pb_p
$$

💬 老師補充：
- **向量其實沒有「乘法」**：乘法的意思是乘完之後還是同樣大小的東西。比較接近的有「外積」（我們目前用不到）和「**內積**」。
- 在數學系修線性代數會發現，「內積」其實有好幾種，**dot product 才是比較精確的說法**。但在本課程中，**內積就是 dot product**，就是我們小時候學過的那種。
- **矩陣相乘的整個過程，就是在做很多很多個不同的內積**——這件事非常重要。

### 2.4 列向量為中心的故事開展（P13–P14）

- 一般說的線性轉換寫成 $A\mathbf{x}$（$\mathbf{x}$ 是**行向量**）。
- 但 **Google 超喜歡列向量（和內積）**，於是引發了後面的故事……
- **矩陣乘法最重要的技巧**：$\mathbf{x}A$ 正好是用 $\mathbf{x}$ 的分量，對 $A$ 的**列向量做線性組合**：

$$
\mathbf{x}A=[x_1,x_2,\dots,x_m]\begin{bmatrix}\mathbf{a}_1\\ \mathbf{a}_2\\ \vdots\\ \mathbf{a}_m\end{bmatrix}=x_1\mathbf{a}_1+x_2\mathbf{a}_2+\cdots+x_m\mathbf{a}_m
$$

💬 老師補充：**線性代數裡最重要、但很少人認真跟你講過的事**
- 黑板上的例子（矩陣乘**行向量**）：

$$
\begin{bmatrix}1&2&1\\1&1&1\end{bmatrix}\begin{bmatrix}1\\1\\0\end{bmatrix}=\begin{bmatrix}3\\2\end{bmatrix}
\quad\text{也剛好等於}\quad
1\begin{bmatrix}1\\1\end{bmatrix}+1\begin{bmatrix}2\\1\end{bmatrix}+0\begin{bmatrix}1\\1\end{bmatrix}=\begin{bmatrix}3\\2\end{bmatrix}
$$

  也就是說，**$A\mathbf{x}$ 就是把 $\mathbf{x}$ 的數字一個個拿出來，對 $A$ 的「行向量」做線性組合**。「你知道這件事，很多事情都沒問題；不知道這件事，很多事情都有問題。」
- 在機器學習、AI 的世界，特別是 **Google 派系，很喜歡把向量寫成列向量**。列向量不能放在矩陣後面乘，只能放在前面，所以變成 $\mathbf{x}A$——意思一樣，只是反過來：**放後面是對行向量做線性組合，放前面就是對列向量做線性組合**。
- 大部分 ML／AI 論文裡，**一個向量沒有特別寫轉置的話，基本上就是列向量**。
- **Linear vs. Affine transformation**：linear transformation 就是乘上一個矩陣（$A\mathbf{x}$ 或 $\mathbf{x}A$）；affine transformation 聽起來比較高級，其實只是**再加上一個常數向量**（就像加上偏值 bias）：$A\mathbf{x}+\mathbf{b}$。

---

## 03. QKV 注意力機制（投影片 P15–P40，影片約 0:34 起）

### 3.1 Transformers 的誕生（P16）

- 2017 年 Google 發表〈**Attention is All You Need**〉（Vaswani et al., NIPS 2017），介紹 Transformer 架構。
- 主要想**取代 RNN**；但從標題可知，（當時）Google 野心更大。

💬 老師補充：
- **為什麼要取代 RNN？** RNN 看完第一個字產生 $h_1$，再把 $h_1$ 和第二個字一起送進去產生 $h_2$……「我連念都覺得很煩」。這是**迴圈式的計算**，要算很久，**沒辦法平行化**。Google 想要輸入 $\mathbf{x}_1,\dots,\mathbf{x}_T$ 後，**一次平行算出** $\mathbf{h}_1,\dots,\mathbf{h}_T$。
- 標題很臭屁：意思是 CNN、RNN 什麼的**都可以說再見**了，唯一留下來「幫它打工」的只有**全連結神經網路**。這篇非常囂張的文章馬上引起很大的震撼。
- Transformer 架構圖現在看會覺得超級可怕，但**今天學完之後再回來看，就會覺得沒什麼了不起**。

### 3.2 Attention 機制：神秘的 QKV 概念（P17–P19）

| 符號 | 名稱 | 角色 |
|---|---|---|
| **Q** | Query | 問題 |
| **K** | Key | 「先備知識」的索引 |
| **V** | Value | 「先備知識」的內容 |

- 給定先備知識 $(\mathbf{k}_1,\mathbf{v}_1),\dots,(\mathbf{k}_T,\mathbf{v}_T)$（來自 $\mathbf{x}_1,\dots,\mathbf{x}_T$），輸入問題 $\mathbf{q}$，經過 $f_\theta$ 得到**適合這個問題的結果 $\mathbf{h}$**。
- $\mathbf{q}_t,\mathbf{k}_t,\mathbf{v}_t$ 都是 $\mathbf{x}_t$ 的一個**特徵代表向量**（P18）。
- 直覺：有一段資訊（例如一段文字 $\mathbf{x}_1,\dots,\mathbf{x}_T$），現在有個 query $\mathbf{q}$ 進來，想依前面的資訊，找出最適合 $\mathbf{q}$ 的代表向量 $\mathbf{h}$（P19）。

💬 老師補充：**為什麼要分 Key 跟 Value？不分可以嗎？**
- 當然可以，K 就是 V、V 就是 K 也行。只是 Google 想要**最一般的情況**，所以分成兩種。
- 同一個字想表示成三種不同的特徵代表向量，當然可以；你要 200 種也可以，只是 Transformer 用了三種，特別取了 query、key、value 三個名字而已。

### 3.3 最後這個向量長這樣（P20）

$$
\mathbf{h}=\alpha_1\mathbf{v}_1+\alpha_2\mathbf{v}_2+\cdots+\alpha_T\mathbf{v}_T
$$

- $\alpha_t$ 是 $\mathbf{q}$ 和 $\mathbf{x}_t$ 的**相關強度**（attention 的強度），基本上**愛怎麼合理的算都可以**。

💬 老師補充：我們很喜歡讓 $\alpha_1+\alpha_2+\cdots+\alpha_T=1$（「當然不會是 87」）。例如 30% 的重心放在第一個字、20% 放在第二個字……這很合理。

### 3.4 計算「注意力」的強度（P21–P24）

- Google 最愛的方法：**內積**

$$
a(\mathbf{q},\mathbf{k}_t)=\langle\mathbf{q},\mathbf{k}_t\rangle=\mathbf{q}\mathbf{k}_t^T=e_t
$$

- 深愛列向量的 Google 把所有 $\mathbf{k}_i$、$\mathbf{v}_i$ 各自疊成矩陣 $K$、$V$（每一列是一個向量）（P22）。
- 所有 attention 強度只要**一次矩陣乘法**算完：$\mathbf{q}K^T=[e_1,e_2,\dots,e_T]$（P23）。
- 再經 **softmax** 就是最後的權重係數 $[\alpha_1,\dots,\alpha_T]$（P24）。

💬 老師補充：**attention 強度有很多種算法**
- 用**相關係數**？可以，但算起來太複雜，還要開根號。
- **訓練一個神經元**去算（兩個向量送進去，加權和、加偏值，出來一個數字）？可以，而且在神經網路的世界這**比較自然**，Google 也發現**效果比較好**。但 Google 不喜歡，因為**寫不出只有矩陣相乘的漂亮式子**。
- 所以 Google 堅持用**內積**。如果知道 cosine similarity：長度都是 1 的兩個向量，內積就是夾角的 cosine，越像越接近 1、方向相反是 −1；內積只是**省掉除以長度**那一步，所以真的可以當成相似度、強度來用。
- $e_1,\dots,e_T$ 加起來不一定等於 1，但要讓一堆數字加起來等於 1，我們已經超拿手了：**做 softmax 就好**。

### 3.5 式子可以很美（P25–P26）

$$
\text{softmax}(\mathbf{q}K^T)\,V=[\alpha_1,\dots,\alpha_T]\begin{bmatrix}\mathbf{v}_1\\ \vdots\\ \mathbf{v}_T\end{bmatrix}=\sum_{t=1}^T\alpha_t\mathbf{v}_t
$$

- 記得：這個乘法就是**列向量的線性組合**（第 02 節的技巧）。

$$
\boxed{\text{Attention}(\mathbf{q},K,V)=\text{softmax}(\mathbf{q}K^T)\,V}
$$

💬 老師補充：Google 堅持要用列向量這樣算，原因就是**最後可以寫成一個漂亮的公式，而且完全可以平行化運算**。不用列向量的寫法，你也可以試試看，就是沒那麼漂亮而已。

### 3.6 Self-Attention（P27–P30，影片約 1:03 起）

- 輸入 $\mathbf{x}_1,\dots,\mathbf{x}_T$，輸出 $\mathbf{h}_1,\dots,\mathbf{h}_T$，用來預測 $\hat{\mathbf{x}}_2,\dots,\hat{\mathbf{x}}_{T+1}$——**記得 Google 是想一次生成！**
- 基本上是每個 $\mathbf{x}_t$ 再增加一個 $\mathbf{q}_t$ 特徵代表向量（P28）。
- **其實就是大家輪流當 q**：$\mathbf{q}_j\to\mathbf{h}_j=\sum_t\alpha_t\mathbf{v}_t$，就是標準 attention 計算（P29）。
- 像 $K, V$ 一樣，把所有 $\mathbf{q}_i$ 疊成矩陣 $Q$，就導出 Google 引以為傲的公式（P30）：

$$
\boxed{\text{Attention}(Q,K,V)=\text{softmax}(QK^T)\,V}
$$

💬 老師補充：
- 把 $Q$ 一列一列疊起來後，公式跟剛剛「**八七分像**」。$QK^T$ 的第 1 列就是 $\mathbf{q}_1$ 算出來的強度、第 2 列是 $\mathbf{q}_2$ 的……因為「前面出列、後面出行」。
- 這篇論文的特色是「**把很簡單的概念說得很複雜**」。想法其實很自然：拿這個時間點的字去跟每個字比較，算出 attention 強度（每個字的重心要放多少），再把這些字的 value 做線性組合。但寫得非常「懸疑」，會讓你看很多次都不知道為什麼要 QKV 三種向量。

### 3.7 One More Thing：為什麼要除以 $\sqrt{d_k}$？（P31–P33，影片約 1:06 起）

原始論文的公式其實是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V,\qquad d_k=\dim(\mathbf{k}_i)
$$

- Google 很尷尬地發現：**用內積算 attention 強度，效果沒有別人（例如訓練一個神經元去算）好**。
- 主因是 **softmax「贏者通吃」**，會讓權重高的比合理權重還要高很多。
- 好在這不難救：除以一個**足夠大、適當的數字**就可以（比如 9487），Google 寫 $\sqrt{d_k}$「基本上只是讓你覺得好有學問……」。

**Softmax 贏者通吃（P32–P33）：**

| 原始分數 | 3.9 | 3.2 | 1 | 0.3 | 1.1 |
|---|---|---|---|---|---|
| 直接 softmax | 61% | 30% | 3% | 2% | 4% |
| 同除以 $\tau=\sqrt{d_k}$ 後（1.74, 1.43, 0.45, 0.13, 0.49） | 40% | 29% | 11% | 8% | 11% |

- 本來感覺只差一點，softmax 後卻差很多；**把數字拉到 0 附近**就輕鬆解決。
- $\tau$ 不一定要是 $\sqrt{d_k}$，可以當成一個 **hyperparameter**。

💬 老師補充：
- 內積的數字可能很大，softmax 又贏者通吃，第一名和第二名本來差不多，做完指數轉換後第一名就被拉得太高。解法**跟以前的 temperature 一樣**：除以一個數字就好。
- 如果論文一開始就決定 $\mathbf{k}$ 是 512 維，直接除以 $\sqrt{512}$ 寫出數字也可以。Google 寫成 $\sqrt{d_k}$「看起來比較有學問」，大家都會問「那個根號 $d_k$ 是什麼？」Google 確實有做論證，但老實說**不一定需要**；你覺得除以 $\sqrt{9487}$ 很適合，就放上去。
- **Hyperparameter（超參數）**：建好神經網路、**還沒開始訓練前就要決定**的參數。例如要幾個隱藏層、每層放幾個神經元，都是 hyperparameter。

### 3.8 各個向量是怎麼來的？（P34–P35，影片約 1:12 起）

- $\mathbf{x}_1,\dots,\mathbf{x}_T$ 是 **word embedding**（一般不用 Word2Vec 等現成的，而是直接加 embedding 層去學）。
- $\mathbf{q}_t=\mathbf{x}_tW^Q,\ \mathbf{k}_t=\mathbf{x}_tW^K,\ \mathbf{v}_t=\mathbf{x}_tW^V$：就是對 $\mathbf{x}_t$ 做**線性轉換**，這些矩陣是**學習來的**。
- 把 $\mathbf{x}_t$ 疊成矩陣 $X$，$Q, K, V$ 都可以**一次算出來**：$Q=XW^Q,\ K=XW^K,\ V=XW^V$。

💬 老師補充：**在 Transformer 的世界，真正訓練的就是 $W^Q, W^K, W^V$ 這三個矩陣**。在線性代數裡，有一個向量要變成另一個向量，最自然的就是線性轉換——乘上一個矩陣就結束了。

💬 老師補充：**embedding 層是怎麼來的？**（影片約 1:36）
- 答案非常驚人：現代訓練大型語言模型時，**一開始輸入的只是 one-hot encoding**。例如編號 1 的字就是 $[1,0,0,\dots]$；第 87 號字就是第 87 個位置是 1、其他都是 0。
- 不是說要輸入 word embedding 才有字的意思嗎？訓練 embedding 需要一個**代理任務**（上週介紹過：用旁邊的字預測中間的字，或反過來）。但大型語言模型**本身的任務「看前面一堆字、預測下一個字」就是很好的代理任務**——要懂字的意思才做得好。
- 所以直接把**第一個隱藏層當成 embedding 層**（例如輸出 512 維），跟整個模型一起訓練；模型預測得好，第一層的輸出就很適合當 embedding。

### 3.9 Multi-head Attention（P36）

- 認真想想，attention 沒理由只有一種，所以可以定義第 $n$ 個 attention：$\text{Attention}(XW_n^Q, XW_n^K, XW_n^V)$。

💬 老師補充：換一組 $W^Q, W^K, W^V$，就換了一組特徵代表向量。理解一句話時，第一種 attention 可能注意**哪些字跟意思有關**，第二種可能注意**文法**，第三種又注意別的事——總之讓 AI 自己去學。**原始論文做了 8 個 head**，再把 8 次的結果**串接（concat）**起來。

### 3.10 Encoder 與 Decoder（P37–P40，影片約 1:15 起）

| 部分 | 組成 | 說明 |
|---|---|---|
| **Encoder** | Multi-Head (Self) Attention → Dense | self-attention 的 q, k, v 都由輸入向量「自家生產」 |
| **Decoder** | Masked Multi-Head (Self) Attention → Multi-Head Attention → Dense | 只有**中間那層不是 self-attention**：**k, v 來自 encoder**，**q 來自前一層的 output** |

- Transformer **輸入有幾個（字／詞向量），輸出就有幾個**（P39）。
- Decoder 也是這樣，只是**開始還沒有的輸入會被 mask 住**（Masked Multi-head Attention，P40）。

💬 老師補充：
- **Self-attention** 的意思就是「自己跟自己做 attention」：同一句話裡的每個字跟其他字做 attention。
- 經過 attention 之後用**全連結層**整理——幾乎所有神經網路最後都是全連結層：「**前面是理解，最後是總結，總結一定做全連結層**」。
- **Decoder 才是生成式 AI 的重心**：decoder 做的就是預測下一個字；**GPT 系列基本上只訓練 decoder**。
- **Masked** 沒什麼了不起：一開始只有第一個字，就算出 $h_1$、預測第二個字；後面還沒有的字就「假裝蓋起來」，用一點數學小技巧讓它們**不參與計算**。就是「前面一堆字去預測下一個字」的模型。

💬 老師補充：**Transformer 總結**（影片約 1:20）
- 如果不想知道太多：它沒有什麼神奇的地方。一串字進來，**看前後文**，決定這個位置的 hidden state（記憶）長什麼樣子——而且**參考了整句話**，所以比以前 RNN 的記憶更好一點。
- 連這個都不想知道的話：就記得 $\mathbf{x}_1,\dots,\mathbf{x}_T$ 進來，用**平行化的運算**同時產生 $\mathbf{h}_1,\dots,\mathbf{h}_T$。
- 以後別人說 Transformer，**你就不會害怕了**。

---

## 04. Position Encoding（投影片 P41–P50，影片約 1:22 起）

### 4.1 為什麼需要 Position Encoding？（P42）

- **RNN 是真的一個字一個字讀**；Transformer 為了**平行化**，一次做 self-attention，**每個字的先後順序並沒有真的被考慮**。
- 所以要把**位置資訊**加入原來的文字 embedding：

$$
\mathbf{x}_t\ (\text{word embedding})+\mathbf{p}_t\ (\text{position encoding})
$$

💬 老師補充：
- 認真想 Transformer 的算法就會發現：**把字的順序打亂，算出來的 $\mathbf{h}$ 還是一樣**——它沒有「第一個字、第二個字」的概念。所以要提示它：「不好意思，你是第一個字。」
- 位置向量**不能太大**，不然會**喧賓奪主**，吃掉原本字的意思（例如 $[1,0,0,\dots]$ 的那個 1 就太大了）。
- 其實**不一定要用 Google 的做法**：只要有系統地讓每個位置產生不同的代表向量就好；模型學了那麼多次，自然知道字這樣放和那樣放，結果會不一樣。「但 Google 的做法很酷炫、沒人看得懂，出去外面講，人家會覺得你好厲害；Google 還畫了一張圖，讓你看了更不懂。」

### 4.2 從進位系統找靈感（P43–P44）

- 觀察任何一個進位數字系統：**越高位數，週期越長、頻率越低**。
  - 例：十進位 9487，個位頻率 1、十位 1/10、百位 1/100、千位 1/1000（注意文字表示是相反的，比較像 [7, 8, 4, 9]）。
- 為了通用，想找一個「奇幻進位系統」：**連續函數、有週期性、數值不要太大**（以免影響原 embedding）→ 很容易打 **sin、cos** 的主意！

💬 老師補充：**為什麼要「連續」的進位系統？**
- 因為每個模型的輸入長度不一樣。像 **ChatGPT 剛開始的時候，輸入只能兩千多個字**（例如 2048），因為它訓練時就是吃兩千多個字；超過的部分**不是「忘了」，而是根本沒有輸入**。後來又變成四千多、六千多……所以需要一個能適應各種長度的連續型進位系統。
- 寫文字時第一個字在最左邊，所以「個位數」應該放最前面：**最左邊更新頻率最快，越往右越慢**。

### 4.3 原論文的 position encoding（P45–P48）

- $\mathbf{p}_t=[p_0,p_1,\dots,p_{d-1}]$，**前面是低位數、後面是高位數**；兩兩一組用同一個頻率：

$$
p_i=\begin{cases}\sin(\omega_k t), & i=2k\\ \cos(\omega_k t), & i=2k+1\end{cases},\qquad \omega_k=\frac{1}{10000^{2k/d}}
$$

- $t$ 是第 $t$ 個字；**位數越高頻率越低**（$\omega_0>\omega_1>\cdots$）。
- 注意：原論文足標通常從 1 開始，但這裡應該從 0 開始才符合式子。

💬 老師補充：這邊說得比較快，因為**你也不一定要這麼做**；只是這樣做，公式一出來大家都覺得超酷炫。畫圖給別人看「沒有任何幫助，只會嚇到人家」。記得重點：**有系統地為每個位置產生不同的向量**就好。

### 4.4 為什麼要加入 cos，只有 sin 不行嗎？（P49–P50）

- 沒人規定一個位元只能用一個數字表示。$(\sin\omega_k t,\cos\omega_k t)$ 剛好是**單位圓上的一點**，可以想成**時鐘上的一個點**：**位元低的時鐘快、位元高的時鐘慢**。
- 「極座標明明是 cos 在前！」——雖然不知 Google 的考量，但 $(\sin\theta,\cos\theta)$ 繞一圈也是單位圓；而且 $\theta=0$ 指向一般時鐘 **12 點方向**，走向也是**順時鐘**。

💬 老師補充：它寫成 sin、cos 就是想**模仿時鐘**。小時候學的極座標是 $(\cos\theta,\sin\theta)$，走**逆時鐘**；要從 12 點走**順時鐘**，就變成 $(\sin\theta,\cos\theta)$。「你堅持要 cos、sin 可不可以？當然可以。」

---

## 05. 讓 Transformers 更穩（投影片 P51–P67，影片約 1:34 起）

💬 老師補充：**這一節上課只簡單帶過**（「雖然有一點點重要，但有機會再為大家介紹」），有興趣的同學可以參考投影片。老師口頭說明的重點：
- 原版 Transformer 有很多**穩定性的做法**，包括 **residual block**——這在 AI 世界非常重要，是**讓神經網路變得很深**時的關鍵。
- 還有 **Layer Normalization**（或 Batch Normalization）：避免某一層的輸出**數值突然變大**，讓神經網路不穩；把每次輸出壓在合理範圍內，讓網路跑得比較穩定。

### 5.1 原版 Transformer 架構還沒說的兩件事（P52）

- **Residual 連結**與 **Layer Normalization**：都是當年讓神經網路**更深、更穩定**的最佳技術（好在現在也差不多）。

### 5.2 ResNet 型的設計（P53–P58）

- 本來某一層的輸出是 $\ell(\mathbf{z})$，現在變成 $\mathbf{z}+\ell(\mathbf{z})$。
- 假設目標（正確答案）是 $f(\mathbf{z})$：
  - 本來希望 $\ell(\mathbf{z})\simeq f(\mathbf{z})$
  - 現在變成 $\mathbf{z}+\ell(\mathbf{z})\simeq f(\mathbf{z})$，也就是 $\ell(\mathbf{z})=f(\mathbf{z})-\mathbf{z}$
- **只需要學之前還沒學到的**：$\mathbf{z}$ 是已經學到的；若 $\mathbf{z}$ 已經很接近正確答案，$\ell(\mathbf{z})$ 就不太需要再學什麼，**重心放在前方**。
- 有 skip 連結的 loss function 地形**平順很多**（Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018）。

### 5.3 Normalization（P59–P62）

**Batch Normalization（P59）**：對一個 batch 的 $x_1,\dots,x_m$ 計算平均值、標準差：

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}},\qquad y=\gamma\hat{x}+\beta
$$

（$\varepsilon$ 預防除以 0；$\gamma,\beta$ 是學習來的）

**Layer Normalization（P60，原版 Transformer 的選擇）**：對**同一個向量** $\mathbf{x}=[x_1,\dots,x_n]$ 的分量計算：

$$
\mu=\frac1n\sum_{i=1}^n x_i,\qquad \sigma=\sqrt{\frac1n\sum_{i=1}^n(x_i-\mu)^2},\qquad
\hat{\mathbf{x}}=\frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\varepsilon}},\qquad \mathbf{y}=\gamma\hat{\mathbf{x}}+\beta
$$

- $\mathbf{x}$ 是向量、$\mu$ 是純量，基本上都是像 **numpy 的 broadcasting** 運算；$\gamma,\beta$ 都是向量（P62）。

### 5.4 更新的模型差在哪？（P63–P65）

- 從 GPT-2 到 Llama 2／3／3.1／3.2 的架構比較：<https://github.com/rasbt/LLMs-from-scratch>
- **Llama 等 LLM 運用了 RMSNorm（P64）**：

$$
\text{RMS}(\mathbf{x})=\sqrt{\frac1n\sum_{i=1}^n x_i^2},\qquad \text{RMSNorm}(\mathbf{x})=\frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}\cdot\gamma\quad(\gamma\ \text{是學來的向量})
$$

- **最近 Meta 說不用做 Normalization！（P65）**〈Transformers without Normalization〉提出 **Dynamic Tanh**：

$$
\text{DyT}(\mathbf{x})=\gamma\cdot\tanh(\alpha\mathbf{x})+\beta\qquad(\alpha\ \text{可設為 0.5；}\gamma,\beta\ \text{是學習來的向量})
$$

### 5.5 Transformer 不只用在文字（P66–P67）

- **圖像**（切成小塊）、聲音、時間序列數據也可以丟進 Transformer。
- **「混入」資訊也很好用**：例如文字生圖的 AI，怎麼把文字的意思混進去？
  - $Q$：隨機產生「天馬行空」的想法
  - $K, V$：來自代表文字「意思」的矩陣 $\chi$
  - 經過 Transformer 得到 $Z$：**混入了文字的意涵**

💬 老師補充（影片約 1:39）：經過剛剛「偉大的數學洗禮」，不用太擔心。數學是希望**有興趣的同學真的知道原理**，就不會道聽塗說，很容易判斷別人說的是真是假。如果很討厭數學、兩節課完全不知道老師在幹嘛，**只要記得重點**：這些都是**平行化**運算出來的；$Q, K, V$ 是訓練三個矩陣算出來的；再用 attention 強度把結果算出來。

---

## 06. 作業：和 AI 討論出 Vibe Coding 程式規格書（投影片 P68–P85，影片約 1:39 起）

### 6.1 作業內容（P69）

想一個有趣的應用，用 Vibe Coding 做出一個 web app，並上傳到 GitHub 讓全世界都看得到！

1. 先和 AI 討論，產生這個 App 的「**規格書**」。
2. 用這個規格書產生一個 **`index.html`** 網頁。
3. 用 **GitHub Pages** 讓全世界看到你的作品！

💬 老師補充：**什麼才叫 Vibe Coding？**
- Vibe Coding 是**完全不看程式、不自己改程式**——可以改，但要**叫 AI 改**，不能自己改。
- 像第一次作業（打造神經網路），你有去看程式、改了一小部分，那就不叫 Vibe Coding，最多叫「**AI 協作**」——「你比 Vibe Coding 更進步」。Vibe Coding 就是「完全耍廢、完全不看內容」。
- 上一次第三節助教課，助教已經介紹過怎麼上傳到 GitHub。

### 6.2 互動網頁的三種技術（P70）

| 技術 | 作用 |
|---|---|
| **HTML** | 網頁的基本語言，決定網頁的**內容** |
| **CSS** | 決定網頁的**排版、字型、顏色** |
| **JavaScript** | 讓網頁能**互動**，所有瀏覽器都懂的程式語言 |

- 可以是單一檔案：**基本上單一檔案的 `.html` 檔就可以**。

💬 老師補充：HTML 跟 Markdown 很接近，只是語法不同；JavaScript 才是真正的程式語言，做遊戲、互動都靠它。比較大的網頁會把 CSS、JavaScript 分成不同檔案，但這次只要**告訴 AI「產生單一的 HTML 檔」**就好，電腦很會做這件事。「一頁式」聽起來很像詐騙，反正就是一個網頁。

### 6.3 Prompt 只有兩個要點（P71）

| 要點 | 說明 |
|---|---|
| **正確的資訊** | 它不知道你腦中的背景——把該給的資料、條件講清楚 |
| **清楚的指引** | 要它做什麼、輸出成什麼樣子、給誰看，講明白 |

### 6.4 範例：呆一布大抽獎（P72–P77）

- 想寫個抽獎轉盤程式 → 其實就是要寫一份很好的「**規格書**」（基本資訊、畫面與流程、抽獎規則……）。「我……寫不出這種東西……」
- **Prompt 必殺技：和 AI 一起寫 prompt！**（P74）
- 可以跟 AI 說（P75）：

> 我們呆一布 AI 想辦一個小活動，最後要抽出一個獎品給幸運得主。希望這個程式是單一的 HTML 檔，**先不要寫程式，問我一些問題，幫我把需求釐清**，或是提供一下建議。

- 來來回回討論，最後再讓 AI 把規格書寫出來！
- 程式大概長這樣（P76）：**`index.html`（檔名一定要叫這個）** ＋ **`assets/`（圖片可以放裡面）**。
- 把 `index.html` 拉到瀏覽器，就看到你的 App！（P77）

💬 老師補充：
- Vibe Coding **最重要的就是規格書要寫得非常仔細**：畫面怎麼呈現、每個角色長什麼樣子、怎麼計分、使用者可以控制什麼……說得越細越好。這對沒寫過程式的同學很難，有經驗的人第一次寫也會想不清楚——所以**先跟 AI 討論**，讓它問你問題、把需求釐清，滿意了再請它寫出規格書。
- 這次**不用像上次丟給三個 AI**，生出一個你滿意的就好；除非生得很爛，再換一個 AI 試試看。
- 自己的圖片可以放進資料夾，**名字隨你取**，但**要告訴 AI**——AI 不會知道你有這個資料夾、裡面有什麼圖片。
- 現場示範：討論時有特別要求「**抽獎名單、獎品清單可以自己訂**」（不要寫死，不然用一次就死了）；還有「不在現場」的按鈕，按了會把這個人劃掉、重抽下一位。

### 6.5 放上 GitHub Pages（P78–P85）

1. 先到 <https://github.com/> 申請帳號。
2. 建立一個新的 Repo（右上角 **+ → New repository**）。
3. 取一個自己喜歡的名字、隨便描述一下、**一開始先有 README**（Add README 打開）。
4. 從 **Add file → Upload files** 上傳 `index.html` 和 `assets`，**記得一定要按 Commit**。
5. 到 **Settings → Pages**，Source 選 Deploy from a branch，**Branch 選 `main`** → Save。
6. 用 `https://你的帳號.github.io/repo名稱/` 就能看到你的網頁！
   - 上課示範：<https://yenlung.github.io/vibe-tools/>

💬 老師補充：
- Repo（專案）名稱**盡量用英文小寫**，比較不會出錯；勾選 README 雖然不是絕對必要，但會簡單很多。
- 上傳時可以全部一起拖拉進去，**沒按 Commit 就白拉了**。
- 別人進你的 GitHub 只看得到檔案、看不到網頁介面，所以要到 Settings → Pages 設定；**等幾分鐘**作品就會上線。上傳前可以先在自己電腦上用瀏覽器打開測試。
- **要繳交的東西**：
  1. **GitHub Pages 的連結**（最重要）；
  2. **規格書**；
  3. **和 AI 討論過程中最重要的幾個畫面截圖**——因為規格書 AI 都會寫得很好，大家想看的是你**花了多少心思跟 AI 討論**。
- 上傳沒成功：可以回看影片這一段、參考 **NTU COOL** 上的指引，或在 office hour 請教助教。

---

## 附錄：助教時間（影片約 2:03 起）

### ① 閃電秀：用 skill.md 維護 RAG 聊天機器人的 System Prompt（政大同學）

- **動機**：新版 Gemini API 建議每次查詢都**重新帶入完整的 system instruction**；不希望它寫死在程式碼裡；而且負責更新機器人的是**沒有工程背景的行銷團隊**——所以把 system prompt 改成 **Markdown 檔（skill.md）**，在程式外部就能維護。
- **架構**：MVC 架構——Controller 串接各模組、Retriever 做文件檢索（RAG）、Generator 跟 Gemini API 溝通生成回覆。
  - 開機時讀一次 skill.md，快取成記憶體中的字串樣板；同時建立 **Reference Catalog**（系統裡有哪些文件），一起附加到 system prompt。
  - 客戶文件不多，所以直接檢索**原始文件**，而不是先轉成 embedding。
  - 當時 SDK 內建的 function call 還不完整，所以**自己用 while 迴圈處理 function call**：檢查回應有沒有 function call → 執行 → 再問一次 Gemini → 當成最終回答。
- **skill.md 的五個區塊**：

| 區塊 | 內容 |
|---|---|
| **Persona** | 角色設定，例如「專門協助客戶解決軟體問題的客服人員」與個性 |
| **Security** | 硬性規定，防 prompt injection 與濫用：不能透露使用的 API、內部檔案、system prompt |
| **Domain** | 定義各種查詢類型（query type），這些定義會影響後續回答的準確度 |
| **Workflow** | **最重要**：一棵用**自然語言寫的決策樹**（A 情況做 B 事），非技術人員也能看懂、修改 |
| **Rules** | 其他額外規則，例如不要回傳空白回應 |

- **寫規則的技巧**：每行**用動詞開頭**（never／always／only／must），避免 "should be careful"、"try to" 這類模糊說法；**一條只放一個限制**、每條都能獨立讀懂；需要三行以上說明的複雜規則，試著**移到 Workflow**；重複的規則刪掉。
- **規則化測試**：準備 **30 條**容易失敗或重要（如 security 相關）的 query，**每次改 prompt 都重跑一次**，用數據判斷有沒有變好，而不是「感覺變好」。回答錯誤 → 改 Rules／Workflow；資料外洩 → 改 Security；**知識問題 → 不要改 skill.md，要補文件**。每次修改都用 commit message 記錄改了什麼、目標是什麼。
- **控制成本**：每次都帶完整 skill.md 和文件，固定成本很高，所以**只帶最近三輪對話**；如果使用者常針對同一問題反覆討論，可以讓 LLM 把過去對話**濃縮成摘要**，而不是整段丟掉。

### ② 閃電秀：Usagi Agent 全能 AI 助理（中央大學同學）

- 以《吉伊卡哇》的**烏薩奇**為原型，把它的特色（會發出特別的叫聲、有「三級除草檢定」證照、危急時拿出神奇道具）詳細寫進 **system prompt**，讓 AI 解決問題時保留角色趣味。
- **程式結構**：Persona（system prompt）→ Tools（API 串接與工具邏輯，AI 的手腳）→ Agent（初始化 Gemini、把人設和工具綁在一起，讓 AI 自動呼叫工具）→ 後端 **FastAPI** → 前端 **Streamlit**（對話視窗與背景圖）。
- **四個工具**：美食雷達（串接開放地圖資料，說「肚子餓」就撈出真實餐廳，並要求搜尋時補上國家和城市名稱）、冒險導航（找觀光景點）、三級除草檢定（輸入植物特徵，回傳圖鑑編號和拔除方法）、百寶袋（求救時用隨機函數抽一個道具）。
- **心得**：把 **System Prompt 和 Tool Calling 結合**，可以做出兼具功能性和「生命力」的 AI 助理。

### ③ 助教課：作業四範例分享——電子報分析器（影片約 2:24 起）

助教用「上次作業四」的六個步驟，示範怎麼和 AI 一起把 prompt 寫好：

| 步驟 | 內容 | 助教的做法與提醒 |
|---|---|---|
| 1 | 用直覺寫 prompt | 「幫我做一個自動整理電子報的分析器」→ 結果只有很一般的摘要，因為 AI 不知道**用途**和**誰要看** |
| 2 | 讓 AI 問你問題 | 用 Claude，它會給 ABCD 選項讓你勾。說清楚用途（存進**個人知識庫**、之後用關鍵字就能搜到）、輸出格式（標題、來源、日期、一句話摘要、**標籤**）、**邊界**（不能加外部資料、不能加作者沒明說的立場） |
| 3 | 補上真實經驗 | 告訴 AI 只有你知道的背景（例如商學院、念過會計）和目的、**成功標準**。只說背景不一定夠，要用強烈字眼**規定必須納入分析** |
| 4 | 檢查哪些要查證 | 請 AI 把專有名詞標成「**待查證**」，查完改成「已確認」；檢查像「吳恩達說 AI Agent 職缺比 FDE 多」到底是原文還是幻覺 |
| 5 | 改寫成新指令 | 難的不是生出 prompt，而是讓它**在不同 AI 上都穩定、少幻覺**：把「什麼是好、什麼是不好」定義清楚，給範例 |
| 6 | 丟給不同 LLM 比較 | 第一版：ChatGPT 最好但口氣呆板；Claude 關鍵詞列得清楚，但把支持者的論點寫成反方（嚴重幻覺）；Gemini 有用上商學院背景，但自己編標題。改成第二版後大多有進步（只有 ChatGPT 分數反而下降） |

**最重要的提醒**：步驟二 AI 問你的問題，**如果你答不出來，就表示你還沒想清楚要請 AI 做什麼**——先想清楚，再回來跟 AI 互動、調整指令。

---

## 名詞對照表

| 名詞 | 說明 |
|---|---|
| RNN（Recurrent Neural Network） | 遞歸神經網路；每個時間點會把上一次的輸出（記憶）一起當輸入 |
| Hidden state | 隱藏狀態，就是神經網路的「記憶」；在某個時間點代表的意義（一串數字） |
| 先列後行 | 矩陣 $m\times n$ 是 $m$ 列 $n$ 行；列（Row）是橫的、行（Column）是直的（與中國大陸、日本用法相反） |
| 內積（dot product） | $\mathbf{u}\mathbf{v}^T=a_1b_1+\cdots+a_pb_p$；矩陣乘法就是做很多內積 |
| 轉置（transpose） | 把矩陣的列變成行，記作 $A^T$ |
| 線性組合 | 各向量乘上倍數後相加；$\mathbf{x}A$ 是對 $A$ 的列向量做線性組合 |
| 線性轉換／仿射轉換 | 乘上一個矩陣（$A\mathbf{x}$）／再加上常數向量（$A\mathbf{x}+\mathbf{b}$） |
| Transformer | 2017 年 Google〈Attention is All You Need〉提出的架構，可平行化運算 |
| Attention（注意力） | 用相關強度對各字的 value 做加權平均，決定每個位置的代表向量 |
| Query／Key／Value | 同一個字的三種特徵代表向量：問題／索引／內容 |
| Self-attention | 同一句話裡的每個字輪流當 query，跟其他字做 attention |
| Softmax | 把一串數字轉成加起來等於 1 的權重；有「贏者通吃」的特性 |
| $\sqrt{d_k}$ | 除以 key 向量維度的平方根，讓 softmax 不要太極端（作用同 temperature） |
| Hyperparameter（超參數） | 訓練前就要先決定的參數，例如隱藏層數、每層神經元數 |
| $W^Q, W^K, W^V$ | 把 $\mathbf{x}_t$ 轉成 q、k、v 的矩陣，是 Transformer 真正要訓練的東西 |
| Multi-head attention | 用多組 $W^Q, W^K, W^V$ 做多種 attention（原論文 8 個 head）再串接 |
| Encoder／Decoder | 編碼器／解碼器；GPT 系列基本上只用 decoder 預測下一個字 |
| Masked attention | 把還沒出現的字蓋起來，不參與計算 |
| One-hot encoding | 只有一個位置是 1、其他是 0 的向量，用來表示字的編號 |
| Embedding 層 | 把 one-hot 轉成有意義向量的第一個隱藏層，跟模型一起訓練 |
| Position encoding | 位置編碼；把每個位置的代表向量加到 word embedding 上 |
| Residual 連結（ResNet） | 輸出變成 $\mathbf{z}+\ell(\mathbf{z})$，只需學還沒學到的部分，讓深層網路好訓練 |
| Batch／Layer Normalization | 把數值標準化到合理範圍，讓訓練穩定；Transformer 原版用 Layer Norm |
| RMSNorm／DyT | Llama 等新模型使用的正規化方法／Meta 提出取代正規化的 Dynamic Tanh |
| Vibe Coding | 完全不看、不自己改程式，只透過和 AI 對話完成程式 |
| 規格書 | 描述 App 功能、畫面、流程、規則的文件，是 Vibe Coding 成敗的關鍵 |
| GitHub Pages | GitHub 的免費網頁託管服務，網址為 `帳號.github.io/repo名稱` |
| skill.md | 用 Markdown 檔維護 system prompt 的做法（Persona、Security、Domain、Workflow、Rules） |
| RAG | 檢索增強生成；先檢索文件，再讓 LLM 根據文件回答 |
| Tool Calling／Function Call | 讓 LLM 呼叫外部工具或 API 的機制 |

---

## 自我檢核

1. RNN 神經元和一般全連結神經元有什麼不同？為什麼說它的「記憶」其實只是數字？
2. 一個 $3\times5$ 的矩陣有幾列、幾行？$a_{23}$ 是哪個位置的元素？台灣的「列／行」和中國大陸的用法有什麼差別？
3. 計算 $\begin{bmatrix}1&2&1\\1&1&1\end{bmatrix}\begin{bmatrix}1\\1\\0\end{bmatrix}$，並用「行向量的線性組合」再算一次，確認答案相同。
4. 為什麼 Google 習慣把向量寫成列向量？這時 $\mathbf{x}A$ 代表對 $A$ 的什麼做線性組合？
5. RNN 最大的缺點是什麼？Transformer 想解決的核心問題是什麼？
6. Q、K、V 各代表什麼？為什麼 Key 和 Value 要分開？不分開可以嗎？
7. attention 強度可以怎麼算？Google 為什麼選擇內積，而不是訓練一個神經元？
8. 請寫出 $\text{Attention}(Q,K,V)$ 的完整公式，並說明 softmax 和 $\sqrt{d_k}$ 各自的作用。
9. 分數 3.9 和 3.2 做 softmax 後變成 61% 和 30%，這說明了 softmax 的什麼特性？$\sqrt{d_k}$ 一定要用嗎？
10. 在 Transformer 裡，真正要訓練的是哪些矩陣？$Q, K, V$ 怎麼一次算出來？
11. Multi-head attention 的直覺是什麼？原論文用了幾個 head？
12. Encoder 和 Decoder 的中間層 attention 有什麼不同？為什麼 decoder 要用 masked attention？GPT 系列主要用哪一部分？
13. 大型語言模型一開始輸入的是 one-hot encoding，那 word embedding 是怎麼來的？
14. 為什麼 Transformer 需要 position encoding？位置向量為什麼不能太大？為什麼需要「連續型」的進位系統？
15. 原論文的 position encoding 用 sin、cos 兩兩一組，可以怎麼用「時鐘」來理解？為什麼是 sin 在前？
16. Residual 連結和 Layer Normalization 各自解決什麼問題？
17. 什麼才算 Vibe Coding？和「AI 協作」有什麼不同？
18. 這次作業要繳交哪三樣東西？把網頁放上 GitHub Pages 的步驟是什麼？
