#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <list>
#include <cmath>
#include <ctime>
#include <deque>
#include <queue>
#include <stack>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <memory>
#include <variant>
#include <optional>
#include <tuple>
#include <format>

enum class OrderType {
    GoodTilCancel,
    FillAndKill
};

enum class Side {
    Buy,
    Sell
};

struct LevelInfo{
    Price price;
    Quantity quantity_;
}

using Price = std::int32_t;
using Quantity = std::uint32_t;
using OrderId = std::uint64_t;

