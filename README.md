# WHU-CSE-InfomationRetrieve
WHU/CSE/信息检索

一个检索epub内容的信息检索引擎，支持topk和布尔查询


整个项目都是基于github上另一个日文搜索引擎项目改的，直接搜应该能搜到


app.py前端也是直接拿来用了，我自己就改了下颜色，需要的自己可以去搜一下


代码基本都是GPT4O改写的，所以我也看不懂


尤其是布尔查询部分，全是gpt写的，所以可能存在一些问题


eva.py也是4O写的，还存在很大问题，要是评价的话索引需要自己先去查一下获取，属于是拿着答案去考试了，还很麻烦，所以懒得改了


项目结构：


dataset文件夹：存放epub文件，想搜什么自己去资源网站下载吧，文件太多懒得上传了


static


templates html和css


index.pkl 索引，第一次运行会自动生成


app.py网页前端运行，仅支持topk查询，因为是直接挪用的


searchfinal.py，应该是完全体吧......


