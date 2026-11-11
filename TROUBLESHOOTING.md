# 故障排查指南

本指南帮助您解决使用《机器学习初学者》课程时常见的问题。如果您在这里找不到解决方案，请查看我们的[Discord讨论](https://aka.ms/foundry/discord)或[提交问题](https://github.com/microsoft/ML-For-Beginners/issues)。

## 目录

- [安装问题](../..)
- [Jupyter Notebook问题](../..)
- [Python包问题](../..)
- [R环境问题](../..)
- [测验应用问题](../..)
- [数据和文件路径问题](../..)
- [常见错误信息](../..)
- [性能问题](../..)
- [环境和配置](../..)

---

## 安装问题

### Python安装

**问题**：`python: command not found`

**解决方案**：
1. 从[python.org](https://www.python.org/downloads/)安装Python 3.8或更高版本
2. 验证安装：`python --version`或`python3 --version`
3. 在macOS/Linux上，可能需要使用`python3`而不是`python`

**问题**：多个Python版本导致冲突

**解决方案**：
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter安装

**问题**：`jupyter: command not found`

**解决方案**：
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**问题**：Jupyter无法在浏览器中启动

**解决方案**：
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R安装

**问题**：R包无法安装

**解决方案**：
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**问题**：IRkernel在Jupyter中不可用

**解决方案**：
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook问题

### 内核问题

**问题**：内核不断崩溃或重启

**解决方案**：
1. 重启内核：`Kernel → Restart`
2. 清除输出并重启：`Kernel → Restart & Clear Output`
3. 检查内存问题（参见[性能问题](../..)）
4. 尝试逐个运行单元格以识别问题代码

**问题**：选择了错误的Python内核

**解决方案**：
1. 检查当前内核：`Kernel → Change Kernel`
2. 选择正确的Python版本
3. 如果内核缺失，请创建：
```bash
python -m ipykernel install --user --name=ml-env
```

**问题**：内核无法启动

**解决方案**：
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook单元格问题

**问题**：单元格正在运行但不显示输出

**解决方案**：
1. 检查单元格是否仍在运行（查看`[*]`指示器）
2. 重启内核并运行所有单元格：`Kernel → Restart & Run All`
3. 检查浏览器控制台是否有JavaScript错误（按F12）

**问题**：无法运行单元格——点击“运行”无响应

**解决方案**：
1. 检查Jupyter服务器是否仍在终端中运行
2. 刷新浏览器页面
3. 关闭并重新打开Notebook
4. 重启Jupyter服务器

---

## Python包问题

### 导入错误

**问题**：`ModuleNotFoundError: No module named 'sklearn'`

**解决方案**：
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**问题**：`ImportError: cannot import name 'X' from 'sklearn'`

**解决方案**：
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### 版本冲突

**问题**：包版本不兼容错误

**解决方案**：
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**问题**：`pip install`因权限错误失败

**解决方案**：
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### 数据加载问题

**问题**：加载CSV文件时出现`FileNotFoundError`

**解决方案**：
```python
import os
# Check current working directory
print(os.getcwd())

# Use relative paths from notebook location
df = pd.read_csv('data/filename.csv')

# Or use absolute paths
df = pd.read_csv('/full/path/to/data/filename.csv')
```

---

## R环境问题

### 包安装

**问题**：包安装因编译错误失败

**解决方案**：
```r
# Install binary version (Windows/macOS)
install.packages("package-name", type = "binary")

# Update R to latest version if packages require it
# Check R version
R.version.string

# Install system dependencies (Linux)
# For Ubuntu/Debian, in terminal:
# sudo apt-get install r-base-dev
```

**问题**：`tidyverse`无法安装

**解决方案**：
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown问题

**问题**：RMarkdown无法渲染

**解决方案**：
```r
# Install/update rmarkdown
install.packages("rmarkdown")

# Install pandoc if needed
install.packages("pandoc")

# For PDF output, install tinytex
install.packages("tinytex")
tinytex::install_tinytex()
```

---

## 测验应用问题

### 构建和安装

**问题**：`npm install`失败

**解决方案**：
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install

# If still fails, try with legacy peer deps
npm install --legacy-peer-deps
```

**问题**：端口8080已被占用

**解决方案**：
```bash
# Use different port
npm run serve -- --port 8081

# Or find and kill process using port 8080
# On Linux/macOS:
lsof -ti:8080 | xargs kill -9

# On Windows:
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### 构建错误

**问题**：`npm run build`失败

**解决方案**：
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**问题**：Linting错误阻止构建

**解决方案**：
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## 数据和文件路径问题

### 路径问题

**问题**：运行Notebook时找不到数据文件

**解决方案**：
1. **始终从包含Notebook的目录运行**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **检查代码中的相对路径**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('data/filename.csv')
   
   # Not from your terminal location
   ```

3. **必要时使用绝对路径**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### 数据文件丢失

**问题**：数据集文件丢失

**解决方案**：
1. 检查数据是否应该在仓库中——大多数数据集都已包含
2. 某些课程可能需要下载数据——请查看课程README
3. 确保您已拉取最新的更改：
   ```bash
   git pull origin main
   ```

---

## 常见错误信息

### 内存错误

**错误**：处理数据时出现`MemoryError`或内核崩溃

**解决方案**：
```python
# Load data in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)

# Or read only needed columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])

# Free memory when done
del large_dataframe
import gc
gc.collect()
```

### 收敛警告

**警告**：`ConvergenceWarning: Maximum number of iterations reached`

**解决方案**：
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 绘图问题

**问题**：Jupyter中不显示图表

**解决方案**：
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**问题**：Seaborn图表显示异常或报错

**解决方案**：
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/编码错误

**问题**：读取文件时出现`UnicodeDecodeError`

**解决方案**：
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## 性能问题

### Notebook执行缓慢

**问题**：Notebook运行速度非常慢

**解决方案**：
1. **重启内核释放内存**：`Kernel → Restart`
2. **关闭未使用的Notebook**以释放资源
3. **使用较小的数据样本进行测试**：
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **分析代码性能**以找到瓶颈：
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### 高内存使用

**问题**：系统内存不足

**解决方案**：
```python
# Check memory usage
df.info(memory_usage='deep')

# Optimize data types
df['column'] = df['column'].astype('int32')  # Instead of int64

# Drop unnecessary columns
df = df[['col1', 'col2']]  # Keep only needed columns

# Process in batches
for batch in np.array_split(df, 10):
    process(batch)
```

---

## 环境和配置

### 虚拟环境问题

**问题**：虚拟环境未激活

**解决方案**：
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Check if activated (should show venv name in prompt)
which python  # Should point to venv python
```

**问题**：包已安装但在Notebook中找不到

**解决方案**：
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git问题

**问题**：无法拉取最新更改——出现合并冲突

**解决方案**：
```bash
# Stash your changes
git stash

# Pull latest
git pull origin main

# Reapply your changes
git stash pop

# If conflicts, resolve manually or:
git checkout --theirs path/to/file  # Take remote version
git checkout --ours path/to/file    # Keep your version
```

### VS Code集成

**问题**：Jupyter Notebook无法在VS Code中打开

**解决方案**：
1. 在VS Code中安装Python扩展
2. 在VS Code中安装Jupyter扩展
3. 选择正确的Python解释器：`Ctrl+Shift+P` → "Python: Select Interpreter"
4. 重启VS Code

---

## 其他资源

- **Discord讨论**：[在#ml-for-beginners频道提问并分享解决方案](https://aka.ms/foundry/discord)
- **Microsoft Learn**：[机器学习初学者模块](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **视频教程**：[YouTube播放列表](https://aka.ms/ml-beginners-videos)
- **问题追踪器**：[报告错误](https://github.com/microsoft/ML-For-Beginners/issues)

---

## 仍有问题？

如果您尝试了上述解决方案但仍然遇到问题：

1. **搜索现有问题**：[GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **查看Discord讨论**：[Discord Discussions](https://aka.ms/foundry/discord)
3. **提交新问题**：包括以下内容：
   - 您的操作系统及版本
   - Python/R版本
   - 错误信息（完整回溯）
   - 重现问题的步骤
   - 您已尝试的解决方法

我们随时为您提供帮助！🚀

---

**免责声明**：  
本文档使用AI翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。原始语言的文档应被视为权威来源。对于关键信息，建议使用专业人工翻译。我们不对因使用此翻译而产生的任何误解或误读承担责任。