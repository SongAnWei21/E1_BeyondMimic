"""Package containing task implementations for various robotic environments."""

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)


# 传统的写法（我们上一条讨论的）需要程序员手动一层一层地写 import（from . import xxx ）。如果团队里有 10 个人在同时添加不同的机器人，每天都要修改同一个 __init__.py，就很容易发生代码冲突。

# 所以 Isaac Lab 引入了 import_packages 这个小工具：

# 当 Python 运行到 import_packages(__name__, ...) 时，它会以当前所在的 tasks 文件夹为起点。

# 像一个爬虫一样，自动向下遍历所有的子文件夹（比如 tracking, locomotion 等）和里面的 .py 文件。

# 只要遇到 .py 文件，它就会自动在后台执行 import 操作。

# 这也就意味着，只要你把代码文件丢进 tasks 文件夹下的任何地方，它就会被自动加载！ 甚至连 __init__.py 都不需要写任何内容。
