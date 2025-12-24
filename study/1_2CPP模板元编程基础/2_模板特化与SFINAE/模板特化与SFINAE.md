# 模板特化与SFINAE

> SFINAE (Substitution Failure Is Not An Error) 是C++模板元编程的核心技术

---

## 1. 模板特化概述

### 1.1 什么是模板特化

模板特化允许我们为**特定类型**提供不同于通用模板的实现：

```cpp
// 通用模板
template <typename T>
class Storage {
    T data;
public:
    void store(T value) { data = value; }
};

// 针对bool的特化：使用位压缩存储
template <>
class Storage<bool> {
    unsigned char bits;
public:
    void store(bool value) { /* 位操作 */ }
};
```

### 1.2 为什么需要特化

```
┌─────────────────────────────────────────────────────────────────┐
│                    模板特化的应用场景                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 性能优化                                                     │
│     - 为特定类型提供更高效的实现                                 │
│     - 例：std::vector<bool> 使用位压缩                          │
│                                                                  │
│  2. 特殊行为                                                     │
│     - 某些类型需要完全不同的处理逻辑                             │
│     - 例：指针类型需要特殊的内存管理                             │
│                                                                  │
│  3. 类型特征（Type Traits）                                      │
│     - 编译期获取类型信息                                         │
│     - 例：std::is_pointer<T> 判断是否为指针                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 全特化（Full Specialization）

### 2.1 函数模板全特化

```cpp
// 通用模板
template <typename T>
T maxValue(T a, T b) {
    return (a > b) ? a : b;
}

// 针对C字符串的全特化
template <>
const char* maxValue<const char*>(const char* a, const char* b) {
    return (strcmp(a, b) > 0) ? a : b;
}

// 使用
maxValue(10, 20);           // 使用通用模板
maxValue("apple", "banana"); // 使用特化版本
```

### 2.2 类模板全特化

```cpp
// 通用模板
template <typename T>
class TypeInfo {
public:
    static const char* name() { return "unknown"; }
    static bool isIntegral() { return false; }
};

// 针对int的全特化
template <>
class TypeInfo<int> {
public:
    static const char* name() { return "int"; }
    static bool isIntegral() { return true; }
};

// 针对double的全特化
template <>
class TypeInfo<double> {
public:
    static const char* name() { return "double"; }
    static bool isIntegral() { return false; }
};
```

### 2.3 全特化语法要点

```cpp
// 通用模板声明
template <typename T, typename U>
class Pair { /* ... */ };

// 全特化：必须指定所有模板参数
template <>                         // 空的template<>
class Pair<int, int> { /* ... */ }; // 完全指定类型
```

---

## 3. 偏特化（Partial Specialization）

### 3.1 什么是偏特化

偏特化只指定**部分**模板参数，或对参数添加**约束**：

```cpp
// 通用模板
template <typename T, typename U>
class Pair {
    T first;
    U second;
};

// 偏特化：当两个类型相同时
template <typename T>
class Pair<T, T> {
    T first;
    T second;
    // 可以添加特殊方法
    T sum() { return first + second; }
};

// 偏特化：当第二个类型是int时
template <typename T>
class Pair<T, int> {
    T first;
    int second;
};
```

### 3.2 指针类型偏特化

```cpp
// 通用模板
template <typename T>
class Container {
    T data;
public:
    void process() {
        std::cout << "Processing value" << std::endl;
    }
};

// 偏特化：针对所有指针类型
template <typename T>
class Container<T*> {
    T* data;
public:
    void process() {
        std::cout << "Processing pointer, dereferencing..." << std::endl;
        if (data) {
            // 可以解引用
        }
    }
};

// 使用
Container<int> c1;    // 使用通用模板
Container<int*> c2;   // 使用指针偏特化
Container<double*> c3; // 使用指针偏特化
```

### 3.3 偏特化图解

```
                    template <typename T, typename U>
                    class Pair { ... };
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Pair<T, T>  │    │ Pair<T, int>│    │ Pair<T*, U> │
    │             │    │             │    │             │
    │ 两类型相同  │    │ 第二个是int │    │ 第一个是指针│
    └─────────────┘    └─────────────┘    └─────────────┘
         偏特化             偏特化             偏特化
```

### 3.4 注意：函数模板不支持偏特化

```cpp
// ❌ 错误：函数模板不能偏特化
template <typename T>
void process(T* ptr) { }  // 这不是偏特化，而是重载！

// ✓ 正确做法：使用重载或类模板包装
template <typename T>
struct Processor {
    static void process(T value) { /* 通用实现 */ }
};

template <typename T>
struct Processor<T*> {
    static void process(T* ptr) { /* 指针特化实现 */ }
};
```

---

## 4. SFINAE原理

### 4.1 什么是SFINAE

**SFINAE = Substitution Failure Is Not An Error**

当编译器尝试用具体类型替换模板参数时，如果产生无效代码，编译器**不会报错**，而是简单地忽略这个模板，继续尝试其他重载。

```cpp
template <typename T>
typename T::value_type getValue(T container) {
    return container[0];
}

template <typename T>
T getValue(T value) {
    return value;
}

// 调用
getValue(std::vector<int>{1, 2, 3}); // 使用第一个模板，返回int
getValue(42);                         // int没有value_type，第一个模板SFINAE失败
                                      // 使用第二个模板
```

### 4.2 SFINAE工作流程

```
调用 getValue(42)
      │
      ▼
尝试模板1: typename T::value_type getValue(T)
      │
      ▼
替换 T = int
      │
      ▼
int::value_type 不存在！
      │
      ▼
SFINAE: 不报错，丢弃此模板
      │
      ▼
尝试模板2: T getValue(T)
      │
      ▼
替换 T = int → 有效！
      │
      ▼
选择模板2
```

### 4.3 SFINAE发生的位置

SFINAE只在**函数签名**的模板参数替换中生效：
- 返回类型
- 参数类型
- 默认模板参数
- 非类型模板参数的表达式

```cpp
// ✓ SFINAE可以在这些位置生效
template <typename T, typename = typename T::value_type>  // 默认模板参数
typename T::iterator                                       // 返回类型
func(typename T::const_reference arg);                     // 参数类型

// ❌ 函数体内的错误不触发SFINAE，会直接报错
template <typename T>
void func(T value) {
    typename T::nonexistent_type x;  // 编译错误，不是SFINAE
}
```

---

## 5. std::enable_if

### 5.1 enable_if原理

`std::enable_if` 是利用SFINAE实现条件编译的核心工具：

```cpp
// 简化实现
template <bool Condition, typename T = void>
struct enable_if {};  // 默认情况：没有type成员

template <typename T>
struct enable_if<true, T> {
    using type = T;   // 条件为true时，定义type成员
};

// 使用
enable_if<true, int>::type;   // = int
enable_if<false, int>::type;  // 编译错误：没有type成员 → SFINAE
```

### 5.2 使用enable_if控制函数重载

```cpp
#include <type_traits>

// 只对整数类型启用
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
process(T value) {
    std::cout << "Processing integer: " << value << std::endl;
    return value * 2;
}

// 只对浮点类型启用
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
process(T value) {
    std::cout << "Processing float: " << value << std::endl;
    return value * 2.0;
}

// 使用
process(42);    // 调用整数版本
process(3.14);  // 调用浮点版本
// process("hello"); // 编译错误：没有匹配的重载
```

### 5.3 C++14简化写法

```cpp
// C++14提供了_t后缀简化
template <typename T>
std::enable_if_t<std::is_integral_v<T>, T>  // 注意：enable_if_t 和 is_integral_v
process(T value) {
    return value * 2;
}
```

### 5.4 enable_if在默认模板参数中使用

```cpp
// 更清晰的写法：在默认模板参数中使用
template <typename T, 
          typename = std::enable_if_t<std::is_integral_v<T>>>
T doubleValue(T value) {
    return value * 2;
}

// 或者使用非类型模板参数
template <typename T,
          std::enable_if_t<std::is_integral_v<T>, int> = 0>
T tripleValue(T value) {
    return value * 3;
}
```

---

## 6. std::void_t与类型检测

### 6.1 void_t原理

`std::void_t` (C++17) 将任意类型映射到 `void`，用于检测类型特征：

```cpp
// 简化实现
template <typename...>
using void_t = void;

// 使用void_t检测成员
template <typename T, typename = void>
struct has_value_type : std::false_type {};

template <typename T>
struct has_value_type<T, std::void_t<typename T::value_type>> 
    : std::true_type {};

// 使用
has_value_type<std::vector<int>>::value;  // true
has_value_type<int>::value;                // false
```

### 6.2 检测成员函数

```cpp
// 检测是否有size()成员函数
template <typename T, typename = void>
struct has_size : std::false_type {};

template <typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

// 使用
has_size<std::vector<int>>::value;  // true
has_size<int>::value;                // false
```

### 6.3 类型检测图解

```
has_value_type<std::vector<int>>
              │
              ▼
尝试特化版本：
has_value_type<T, std::void_t<typename T::value_type>>
              │
              ▼
T = std::vector<int>
std::vector<int>::value_type = int  ✓ 存在
              │
              ▼
void_t<int> = void
              │
              ▼
匹配特化版本 → true_type


has_value_type<int>
              │
              ▼
尝试特化版本：
has_value_type<T, std::void_t<typename T::value_type>>
              │
              ▼
T = int
int::value_type  ✗ 不存在
              │
              ▼
SFINAE失败，回退到主模板 → false_type
```

---

## 7. 实际应用示例

### 7.1 类型安全的序列化

```cpp
#include <type_traits>
#include <iostream>
#include <vector>
#include <string>

// 检测是否有serialize成员函数
template <typename T, typename = void>
struct is_serializable : std::false_type {};

template <typename T>
struct is_serializable<T, std::void_t<
    decltype(std::declval<T>().serialize())
>> : std::true_type {};

// 有serialize方法的类型
template <typename T>
std::enable_if_t<is_serializable<T>::value, std::string>
toJson(const T& obj) {
    return obj.serialize();
}

// 基本类型
template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::string>
toJson(T value) {
    return std::to_string(value);
}

// 字符串类型
std::string toJson(const std::string& str) {
    return "\"" + str + "\"";
}
```

### 7.2 编译期类型分发

```cpp
template <typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integer: " << value << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Float: " << value << std::endl;
    } else if constexpr (std::is_pointer_v<T>) {
        std::cout << "Pointer: " << *value << std::endl;
    } else {
        std::cout << "Other type" << std::endl;
    }
}
```

---

## 8. 与FlashAttention的关联

### 8.1 static_switch.h中的SFINAE

FlashAttention使用宏和模板实现编译期分支：

```cpp
// 来自 csrc/flash_attn/src/static_switch.h

// 简化版BOOL_SWITCH宏
#define BOOL_SWITCH(COND, CONST_NAME, ...)                 \
    [&] {                                                   \
        if (COND) {                                         \
            constexpr static bool CONST_NAME = true;        \
            return __VA_ARGS__();                           \
        } else {                                            \
            constexpr static bool CONST_NAME = false;       \
            return __VA_ARGS__();                           \
        }                                                   \
    }()

// 使用示例
BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    // Is_causal 是编译期常量
    // 编译器可以消除无用分支
    run_kernel<Is_causal>(...);
});
```

### 8.2 HeadDim分发

```cpp
// 简化版HEADDIM_SWITCH
#define HEADDIM_SWITCH(HEADDIM, ...)                        \
    [&] {                                                    \
        if (HEADDIM <= 32) {                                 \
            constexpr static int kHeadDim = 32;              \
            return __VA_ARGS__();                            \
        } else if (HEADDIM <= 64) {                          \
            constexpr static int kHeadDim = 64;              \
            return __VA_ARGS__();                            \
        } else if (HEADDIM <= 128) {                         \
            constexpr static int kHeadDim = 128;             \
            return __VA_ARGS__();                            \
        } else {                                             \
            constexpr static int kHeadDim = 256;             \
            return __VA_ARGS__();                            \
        }                                                    \
    }()

// 使用
HEADDIM_SWITCH(params.d, [&] {
    // kHeadDim 是编译期常量，允许优化
    run_mha_fwd_<elem_type, kHeadDim>(params, stream);
});
```

### 8.3 类型特化在kernel_traits中的应用

```cpp
// kernel配置根据不同HeadDim使用不同参数
template <int kHeadDim>
struct HeadDimTraits;

template <>
struct HeadDimTraits<64> {
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 64;
    static constexpr int kNWarps = 4;
};

template <>
struct HeadDimTraits<128> {
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;
    static constexpr int kNWarps = 4;
};
```

---

## 9. 总结

| 技术 | 用途 | 示例 |
|------|------|------|
| 全特化 | 为特定类型提供完全不同的实现 | `template<> class C<int>` |
| 偏特化 | 为类型模式提供特殊实现 | `template<typename T> class C<T*>` |
| SFINAE | 编译期条件选择重载 | 替换失败时忽略模板 |
| enable_if | 显式控制模板启用条件 | `enable_if_t<is_integral_v<T>>` |
| void_t | 检测类型特征 | `void_t<typename T::type>` |

### 选择指南

```
需要为特定类型完全不同的实现？
    → 使用全特化

需要为一类类型（如所有指针）提供不同实现？
    → 使用偏特化（类模板）或重载（函数模板）

需要根据类型特征条件启用/禁用？
    → 使用SFINAE + enable_if

需要检测类型是否有某特征？
    → 使用void_t + 特化
```

---

## 📚 延伸阅读

- [cppreference - SFINAE](https://en.cppreference.com/w/cpp/language/sfinae)
- [cppreference - std::enable_if](https://en.cppreference.com/w/cpp/types/enable_if)
- [cppreference - std::void_t](https://en.cppreference.com/w/cpp/types/void_t)

