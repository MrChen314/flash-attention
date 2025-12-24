# constexpr编译期计算

> 将运行时计算提前到编译期，实现零运行时开销的优化

---

## 1. constexpr概述

### 1.1 什么是constexpr

`constexpr` 是C++11引入的关键字，用于声明**可在编译期求值**的常量或函数。

```cpp
constexpr int size = 100;           // 编译期常量
constexpr int square(int x) {       // 编译期函数
    return x * x;
}

int arr[square(10)];  // OK: 数组大小在编译期确定为100
```

### 1.2 constexpr vs const

```cpp
const int a = 10;              // 运行期常量（可能被优化为编译期）
constexpr int b = 10;          // 明确要求是编译期常量

const int c = getValue();      // OK: 运行期确定
constexpr int d = getValue();  // 错误: getValue()不是constexpr
```

### 1.3 为什么使用constexpr

```
┌─────────────────────────────────────────────────────────────────┐
│                    constexpr的优势                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 零运行时开销                                                 │
│     - 计算在编译期完成，运行时直接使用结果                        │
│                                                                  │
│  2. 类型安全                                                     │
│     - 比宏定义更安全，有类型检查                                  │
│     - 比宏定义更易调试                                           │
│                                                                  │
│  3. 可用于模板参数                                               │
│     - 数组大小、非类型模板参数必须是编译期常量                    │
│                                                                  │
│  4. 编译器优化                                                   │
│     - 编译器可以内联、展开使用constexpr的代码                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. constexpr变量

### 2.1 基本用法

```cpp
// 字面量类型的constexpr变量
constexpr int maxSize = 1024;
constexpr double pi = 3.14159265358979;
constexpr char newline = '\n';

// constexpr数组
constexpr int primes[] = {2, 3, 5, 7, 11, 13};
constexpr size_t numPrimes = sizeof(primes) / sizeof(primes[0]);

// constexpr指针（指向静态存储的地址）
static constexpr int value = 42;
constexpr const int* ptr = &value;
```

### 2.2 字面量类型要求

constexpr变量必须是**字面量类型（Literal Type）**：
- 标量类型（int, float, pointer等）
- 引用类型
- 字面量类型的数组
- 满足特定条件的类类型

```cpp
// ✓ 字面量类型
constexpr int a = 10;
constexpr double b = 3.14;
constexpr int arr[3] = {1, 2, 3};

// ✗ 非字面量类型（std::string有动态分配）
// constexpr std::string s = "hello";  // C++17前错误

// C++20起std::string在某些情况下可以是constexpr
```

---

## 3. constexpr函数

### 3.1 C++11的限制

C++11中constexpr函数限制很严格：

```cpp
// C++11: 只能有一个return语句
constexpr int factorial_11(int n) {
    return n <= 1 ? 1 : n * factorial_11(n - 1);
}

// 不能有：
// - 局部变量
// - 循环
// - 多个return语句
```

### 3.2 C++14的放松

C++14大幅放松了限制：

```cpp
// C++14: 可以有局部变量、循环、多个return
constexpr int factorial_14(int n) {
    if (n <= 1) return 1;
    
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}
```

### 3.3 C++17/20进一步扩展

```cpp
// C++17: constexpr lambda
constexpr auto square = [](int x) { return x * x; };

// C++20: constexpr虚函数、try-catch、动态分配等
constexpr int allocExample() {
    int* p = new int(42);  // C++20: constexpr new
    int result = *p;
    delete p;              // 必须在同一函数中delete
    return result;
}
```

### 3.4 constexpr函数的双重性质

constexpr函数可以在**编译期**或**运行期**调用：

```cpp
constexpr int square(int x) { return x * x; }

// 编译期调用
constexpr int a = square(10);    // 编译期计算
static_assert(square(5) == 25);  // 编译期断言

// 运行期调用
int x;
std::cin >> x;
int b = square(x);  // 运行期计算（x不是编译期常量）
```

---

## 4. constexpr类

### 4.1 字面量类的要求

类要成为字面量类型需满足：
1. 有constexpr构造函数
2. 析构函数是trivial的
3. 所有非静态成员是字面量类型

```cpp
class Point {
    int x_, y_;
public:
    constexpr Point(int x, int y) : x_(x), y_(y) {}
    
    constexpr int x() const { return x_; }
    constexpr int y() const { return y_; }
    
    constexpr int distanceSquared() const {
        return x_ * x_ + y_ * y_;
    }
};

// 编译期使用
constexpr Point origin(0, 0);
constexpr Point p(3, 4);
constexpr int dist = p.distanceSquared();  // 25

static_assert(dist == 25, "Distance should be 25");
```

### 4.2 constexpr成员函数

```cpp
class Rectangle {
    int width_, height_;
public:
    constexpr Rectangle(int w, int h) : width_(w), height_(h) {}
    
    constexpr int width() const { return width_; }
    constexpr int height() const { return height_; }
    constexpr int area() const { return width_ * height_; }
    constexpr int perimeter() const { return 2 * (width_ + height_); }
    
    // C++14起可以修改成员
    constexpr void scale(int factor) {
        width_ *= factor;
        height_ *= factor;
    }
};

constexpr int calculateArea() {
    Rectangle rect(10, 20);
    rect.scale(2);          // 编译期修改
    return rect.area();     // 返回800
}

static_assert(calculateArea() == 800);
```

---

## 5. 编译期数学运算

### 5.1 常用数学函数

```cpp
// 编译期绝对值
constexpr int abs(int x) {
    return x < 0 ? -x : x;
}

// 编译期最大公约数
constexpr int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

// 编译期幂运算
constexpr long long power(int base, int exp) {
    long long result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

// 编译期平方根（牛顿迭代法）
constexpr double sqrt_newton(double x, double guess = 1.0) {
    double new_guess = (guess + x / guess) / 2.0;
    // 精度足够时返回
    return (new_guess - guess < 0.0001 && guess - new_guess < 0.0001)
           ? new_guess
           : sqrt_newton(x, new_guess);
}
```

### 5.2 编译期查找表

```cpp
// 编译期生成查找表
template <size_t N>
struct SinTable {
    double values[N];
    
    constexpr SinTable() : values{} {
        for (size_t i = 0; i < N; ++i) {
            double angle = static_cast<double>(i) / N * 2 * 3.14159265358979;
            values[i] = sin_taylor(angle);  // 使用泰勒展开
        }
    }
    
private:
    static constexpr double sin_taylor(double x) {
        double result = x;
        double term = x;
        for (int i = 1; i < 10; ++i) {
            term *= -x * x / ((2 * i) * (2 * i + 1));
            result += term;
        }
        return result;
    }
};

constexpr SinTable<360> sinLookup;  // 编译期生成360个sin值
```

---

## 6. constexpr与模板

### 6.1 用于非类型模板参数

```cpp
template <int N>
struct Array {
    int data[N];
    constexpr int size() const { return N; }
};

constexpr int calculateSize() {
    return 64 + 32;  // 编译期计算
}

Array<calculateSize()> arr;  // Array<96>
```

### 6.2 编译期条件配置

```cpp
// 类似FlashAttention中的kernel配置
template <int HeadDim>
struct KernelConfig {
    static constexpr int kBlockM = HeadDim <= 64 ? 128 : 64;
    static constexpr int kBlockN = 64;
    static constexpr int kNWarps = HeadDim <= 64 ? 4 : 8;
    
    // 派生常量
    static constexpr int kBlockElements = kBlockM * kBlockN;
    static constexpr int kThreadsPerBlock = kNWarps * 32;
    
    // 编译期验证
    static_assert(kBlockM % 32 == 0, "BlockM must be multiple of 32");
    static_assert(kBlockElements <= 16384, "Block too large for shared memory");
};

// 使用
using Config64 = KernelConfig<64>;
using Config128 = KernelConfig<128>;

static_assert(Config64::kBlockM == 128);
static_assert(Config128::kBlockM == 64);
```

---

## 7. 与FlashAttention的关联

### 7.1 编译期常量定义

FlashAttention大量使用constexpr定义编译期常量：

```cpp
// 来自 kernel_traits.h 的简化版本
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Flash_fwd_kernel_traits {
    // 编译期常量
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kNWarps = kNWarps_;
    
    // 派生的编译期常量
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    
    // 共享内存大小计算（编译期）
    static constexpr int kSmemQSize = kBlockM * kHeadDim * sizeof(float);
    static constexpr int kSmemKVSize = kBlockN * kHeadDim * sizeof(float) * 2;
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;
};
```

### 7.2 循环边界优化

编译期已知的循环边界允许编译器进行循环展开：

```cpp
// 当kBlockN是编译期常量时，编译器可以展开循环
template <int kBlockN>
__device__ void processBlock(float* data) {
    #pragma unroll  // 编译器提示
    for (int i = 0; i < kBlockN; ++i) {  // kBlockN编译期已知
        data[i] *= 2.0f;
    }
}
```

### 7.3 编译期选择最优配置

```cpp
// 根据HeadDim在编译期选择最优的block配置
template <int HeadDim>
constexpr auto selectBlockConfig() {
    if constexpr (HeadDim <= 32) {
        return std::make_pair(128, 128);  // BlockM, BlockN
    } else if constexpr (HeadDim <= 64) {
        return std::make_pair(128, 64);
    } else if constexpr (HeadDim <= 128) {
        return std::make_pair(64, 64);
    } else {
        return std::make_pair(64, 32);
    }
}

// 使用
constexpr auto config = selectBlockConfig<64>();
constexpr int kBlockM = config.first;   // 128
constexpr int kBlockN = config.second;  // 64
```

---

## 8. 最佳实践

### 8.1 何时使用constexpr

| 场景 | 是否使用constexpr | 原因 |
|------|-------------------|------|
| 数组大小 | ✓ | 必须是编译期常量 |
| 模板参数 | ✓ | 必须是编译期常量 |
| 性能关键的计算 | ✓ | 零运行时开销 |
| 配置参数 | ✓ | 允许编译器优化 |
| 复杂的运行时计算 | ✗ | constexpr增加编译时间 |
| 依赖运行时输入 | ✗ | 无法在编译期确定 |

### 8.2 调试技巧

```cpp
// 使用static_assert验证编译期计算
constexpr int result = complexCalculation();
static_assert(result == expectedValue, "Calculation error!");

// 强制编译期求值
template <auto V>
constexpr auto force_constexpr = V;

constexpr int val = force_constexpr<myFunction()>;  // 必须编译期求值
```

---

## 9. 总结

| 特性 | C++11 | C++14 | C++17 | C++20 |
|------|-------|-------|-------|-------|
| constexpr变量 | ✓ | ✓ | ✓ | ✓ |
| constexpr函数（基础） | ✓ | ✓ | ✓ | ✓ |
| 局部变量/循环 | ✗ | ✓ | ✓ | ✓ |
| constexpr lambda | ✗ | ✗ | ✓ | ✓ |
| constexpr虚函数 | ✗ | ✗ | ✗ | ✓ |
| constexpr动态分配 | ✗ | ✗ | ✗ | ✓ |

### 关键要点

1. **constexpr变量**必须用编译期常量初始化
2. **constexpr函数**可以在编译期或运行期调用
3. C++14后constexpr函数限制大幅放松
4. 与模板结合使用可实现强大的编译期计算
5. FlashAttention利用constexpr实现零开销的kernel配置

---

## 📚 延伸阅读

- [cppreference - constexpr](https://en.cppreference.com/w/cpp/language/constexpr)
- [C++17 constexpr if](https://en.cppreference.com/w/cpp/language/if#Constexpr_if)
- [Effective Modern C++ Item 15: Use constexpr whenever possible](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/)

