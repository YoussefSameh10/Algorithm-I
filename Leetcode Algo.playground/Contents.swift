import UIKit

public class ListNode {
    public var val: Int
    public var next: ListNode?
    public init() { self.val = 0; self.next = nil; }
    public init(_ val: Int) { self.val = val; self.next = nil; }
    public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
}

public class TreeNode {
    public var val: Int
    public var left: TreeNode?
    public var right: TreeNode?
    public init() { self.val = 0; self.left = nil; self.right = nil; }
    public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
    public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
        self.val = val
        self.left = left
        self.right = right
    }
}

public class Node {
    public var val: Int
    public var left: Node?
    public var right: Node?
    public var next: Node?
    public init(_ val: Int) {
        self.val = val
        self.left = nil
        self.right = nil
        self.next = nil
    }
}

struct MatrixNode: Comparable {
    var x: Int
    var y: Int
    var val: Int = Int.max
    static func < (lhs: MatrixNode, rhs: MatrixNode) -> Bool {
        lhs.val < rhs.val
    }
    static func > (lhs: MatrixNode, rhs: MatrixNode) -> Bool {
        lhs.val > rhs.val
    }
}

class Stack<T> {
    private var list = [T]()
    private var index = -1
    
    func push(_ element: T) {
        index += 1
        list.append(element)
    }
    
    func pop() -> T {
        let element = list.remove(at: index)
        index -= 1
        return element
    }
    
    func peek() -> T {
        return list[index]
    }
    
    func isEmpty() -> Bool {
        return list.isEmpty
    }
}

class Queue<T> {
    var list = [T]()
    private var index = 0
    
    func push(_ element: T) {
        list.append(element)
    }
    
    func pop() -> T? {
        if index >= list.count {
            return nil
        }
        let element = list.remove(at: index)
        return element
    }
    
    func peek() -> T? {
        if index >= list.count {
            return nil
        }
        return list[index]
    }
    
    func isEmpty() -> Bool {
        return list.isEmpty
    }
}


class Solution {
    // MARK: Algorithm I
    func search(_ nums: [Int], _ target: Int) -> Int {
        let targetIndex = -1
        
        var l = 0, r = nums.count - 1
        while l <= r {
            let m = l + (r - l)/2
            if nums[m] < target {
                l = m + 1
            } else if nums[m] > target {
                r = m - 1
            } else {
                return m
            }
        }
        return targetIndex
    }
    
    private func isBadVersion(_ val: Int) -> Bool {
        return val >= 2
    }
    
    func firstBadVersion(_ n: Int) -> Int {
        var m = -1
        
        var l = 1, r = n
        while l <= r {
            m = l + (r - l)/2
            if !isBadVersion(m) {
                l = m + 1
            } else {
                r = m - 1
            }
        }
        return isBadVersion(m) ? m : m + 1
    }
    
    func searchInsert(_ nums: [Int], _ target: Int) -> Int {
        var m = -1
        
        var l = 0, r = nums.count - 1
        while l <= r {
            m = l + (r - l)/2
            if nums[m] < target {
                l = m + 1
            } else if nums[m] > target {
                r = m - 1
            } else {
                return m
            }
        }
        return nums[m] > target ? m : m + 1
    }
    
    func sortedSquares(_ nums: [Int]) -> [Int] {
        var l = 0, r = nums.count - 1
        var m = -1
        while l <= r {
            m = l + (r - l) / 2
            if nums[m] < 0 {
                l = m + 1
            } else {
                r = m - 1
            }
        }
        m = nums[m] < 0 ? m + 1 : m
        
        r = m
        l = m - 1
        var result = [Int]()
        while l >= 0 && r < nums.count {
            if r >= nums.count || abs(nums[l]) < abs(nums[r]) {
                result.append(nums[l] * nums[l])
                l-=1
            } else {
                result.append(nums[r] * nums[r])
                r+=1
            }
        }
        
        while l >= 0 {
            result.append(nums[l] * nums[l])
            l-=1
        }
        while r < nums.count {
            result.append(nums[r] * nums[r])
            r+=1
        }
        return result
    }
    
    func rotate(_ nums: inout [Int], _ k: Int) {
        var map: [Int: Int] = [:]
        var counter = k%nums.count
        for num in nums {
            map[counter] = num
            counter = (counter + 1)%nums.count
        }
        
        for i in nums.indices {
            nums[i] = map[i] ?? 0
        }
    }
    
    func moveZeroes(_ nums: inout [Int]) {
        for i in nums.indices {
            for j in nums.indices.dropLast(i + 1) {
                if nums[j] == 0 {
                    let temp = nums[j + 1]
                    nums[j + 1] = nums[j]
                    nums[j] = temp
                }
            }
        }
    }
    
    func twoSum(_ numbers: [Int], _ target: Int) -> [Int] {
        var l = 0, r = numbers.count - 1
        
        while l <= r {
            if numbers[l] + numbers[r] < target {
                l+=1
            } else if numbers[l] + numbers[r] > target {
                r-=1
            } else {
                break
            }
        }
        return [l+1, r+1]
    }
    
    func reverseString(_ s: inout [Character]) {
        var l = 0, r = s.count - 1
        
        while l < r {
            let temp = s[l]
            s[l] = s[r]
            s[r] = temp
            l+=1
            r-=1
        }
    }
    
    func reverseWords(_ s: String) -> String {
        let words = s.split(separator: " ")
        var result = ""
        for word in words {
            var characters = [Character]()
            for character in word {
                characters.append(character)
            }
            reverseString(&characters)
            result.append(contentsOf: characters)
            result.append(" ")
        }
        let res = result.dropLast(1)
        return String(res)
    }
    
    func middleNode(_ head: ListNode?) -> ListNode? {
        var counter = 1
        var pointer = head
        while pointer?.next != nil {
            counter+=1
            pointer = pointer?.next
        }
        
        let target: Int = counter/2 + 1
        counter = 1
        var res = head
        while counter < target {
            res = res?.next
            counter+=1
        }
        return res
    }
    
    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        var counter = 1
        var myHead = head
        var pointer = myHead
        while pointer?.next != nil {
            counter+=1
            pointer = pointer?.next
        }
        
        let target: Int = counter - n
        
        if target == 0 {
            myHead = myHead?.next
            return myHead
        } else if target == 1 {
            myHead?.next = myHead?.next?.next
            return myHead
        }
        
        counter = 1
        pointer = myHead
        while counter < target {
            pointer = pointer?.next
            counter+=1
        }
        
        pointer?.next = pointer?.next?.next
        
        return myHead
    }
    
    func lengthOfLongestSubstring(_ s: String) -> Int {
        var i = 0, res = 0, lastRepPos = -1
        var map: [Character: Int] = [:]
        
        for c in s {
            if map[c] != nil {
                lastRepPos = max(lastRepPos, map[c]!)
            }
            res = max(res, i - lastRepPos)
            map[c] = i
            i+=1
        }
        return res
    }
    
    func checkInclusion(_ s1: String, _ s2: String) -> Bool {
        if s2.count < s1.count {
            return false
        }
        var correctMap = [Int](repeating: 0, count: 26)
        var map = [Int](repeating: 0, count: 26)

        for i in s1.indices {
            correctMap[index(for: s1[i])] += 1
            map[index(for: s2[i])] += 1
        }
        
        var count = 0
        for i in map.indices {
            if correctMap[i] == map[i] {
                count += 1
            }
        }
        
        for i in s1.count..<s2.count {
            if count == 26 {
                return true
            }
            
            let old = s2.index(s2.startIndex, offsetBy: i - s1.count)
            let new = s2.index(s2.startIndex, offsetBy: i)
            
            map[index(for: s2[new])] += 1
            if map[index(for: s2[new])] == correctMap[index(for: s2[new])] {
                count += 1
            } else if map[index(for: s2[new])] == correctMap[index(for: s2[new])] + 1 {
                count -= 1
            }
            
            map[index(for: s2[old])] -= 1

            if map[index(for: s2[old])] == correctMap[index(for: s2[old])] {
                count += 1
            } else if map[index(for: s2[old])] == correctMap[index(for: s2[old])] - 1 {
                count -= 1
            }
            
        }
        
        return count == 26
    }
    
    private func index(for char: Character) -> Int {
        Int(char.asciiValue!) % 26
    }
    
    func floodFill(_ image: [[Int]], _ sr: Int, _ sc: Int, _ color: Int) -> [[Int]] {
        var result = image
        let stack = Stack<(x: Int, y: Int)>()
        let originColor = image[sr][sc]
        stack.push((sr, sc))
        if originColor == color {
            return result
        }
        while !stack.isEmpty() {
            let currentPixel = stack.pop()
            if currentPixel.x - 1 >= 0 &&
                result[currentPixel.x - 1][currentPixel.y] != color &&
                result[currentPixel.x - 1][currentPixel.y] == result[currentPixel.x][currentPixel.y] {
                stack.push((x: currentPixel.x - 1, y: currentPixel.y))
            }
            if currentPixel.x + 1 < result.count &&
                result[currentPixel.x + 1][currentPixel.y] != color &&
                result[currentPixel.x + 1][currentPixel.y] == result[currentPixel.x][currentPixel.y] {
                stack.push((x: currentPixel.x + 1, y: currentPixel.y))
            }
            if currentPixel.y - 1 >= 0 &&
                result[currentPixel.x][currentPixel.y - 1] != color &&
                result[currentPixel.x][currentPixel.y - 1] == result[currentPixel.x][currentPixel.y] {
                stack.push((x: currentPixel.x, y: currentPixel.y - 1))
            }
            if currentPixel.y + 1 < result[0].count &&
                result[currentPixel.x][currentPixel.y + 1] != color &&
                result[currentPixel.x][currentPixel.y + 1] == result[currentPixel.x][currentPixel.y] {
                stack.push((x: currentPixel.x, y: currentPixel.y + 1))
            }
            result[currentPixel.x][currentPixel.y] = color
            
        }
        
        return result
    }
    
    func maxAreaOfIsland(_ grid: [[Int]]) -> Int {
        var maximum = 0, current = 0
        let stack = Stack<(x: Int, y: Int)>()
        var land = grid
    
        for i in grid.indices {
            for j in grid[i].indices {
                
                if land[i][j] == 1 {
                    stack.push((x: i, y: j))
                    
                    while !stack.isEmpty() {
                        let currentPosition = stack.pop()
                        
                        if land[currentPosition.x][currentPosition.y] == 1 {
                            land[currentPosition.x][currentPosition.y] = 2
                            current += 1
                        }
                        
                        if currentPosition.x - 1 >= 0 && land[currentPosition.x - 1][currentPosition.y] == 1 {
                            stack.push((x: currentPosition.x - 1, y: currentPosition.y))
                        }
                        if currentPosition.x + 1 < land.count && land[currentPosition.x + 1][currentPosition.y] == 1 {
                            stack.push((x: currentPosition.x + 1, y: currentPosition.y))
                        }
                        if currentPosition.y - 1 >= 0 && land[currentPosition.x][currentPosition.y - 1] == 1 {
                            stack.push((x: currentPosition.x, y: currentPosition.y - 1))
                        }
                        if currentPosition.y + 1 < land[currentPosition.x].count && land[currentPosition.x][currentPosition.y + 1] == 1 {
                            stack.push((x: currentPosition.x, y: currentPosition.y + 1))
                        }
                    }
                    maximum = max(maximum, current)
                    current = 0
                }
            }
        }
        return maximum
    }
        
    func mergeTrees(_ root1: TreeNode?, _ root2: TreeNode?) -> TreeNode? {
        var result: TreeNode?
        
        if root1 == nil {
            return root2
        } else if root2 == nil {
            return root1
        }
        
        result = TreeNode(root1!.val + root2!.val)
        result?.left = mergeTrees(root1?.left, root2?.left)
        result?.right = mergeTrees(root1?.right, root2?.right)
        return result
    }
    
    func connect(_ root: Node?) -> Node? {
        let queue = Queue<Node>()
        var result: Node
        if root == nil {
            return nil
        }
        result = root!
        result.next = nil
        
        queue.push(result)
        
        func isPowerOf2(_ n: Int) -> Bool {
            return (n > 0) && (n & (n - 1) == 0)
        }
        
        while !queue.isEmpty() {
            let pointer = queue.pop()
            if let left = pointer?.left {
                queue.push(left)
            }
            if let right = pointer?.right {
                queue.push(right)
            }
            
            if isPowerOf2(queue.list.count) && queue.list.count > 1 {
                for i in 0..<queue.list.count - 1 {
                    queue.list[i].next = queue.list[i + 1]
                }
            }
        }
        
        return result
    }
    
    func updateMatrix(_ mat: [[Int]]) -> [[Int]] {
        var res = mat
        
        res = mat.map { row in
            row.map { num in
                num != 0 ? Int.max : 0
            }
        }
        
        for i in 0..<res.count {
            for j in 0..<res[i].count {
                if res[i][j] <= 1 {
                    continue
                }
                var neighbours = [MatrixNode(x: i, y: j, val: res[i][j])]
                if i < res.count - 1 {
                    neighbours.append(MatrixNode(x: i + 1, y: j, val: res[i + 1][j]))
                }
                if i > 0 {
                    neighbours.append(MatrixNode(x: i - 1, y: j, val: res[i - 1][j]))
                }
                if j < res[i].count - 1 {
                    neighbours.append(MatrixNode(x: i, y: j + 1, val: res[i][j + 1]))
                }
                if j > 0 {
                    neighbours.append(MatrixNode(x: i, y: j - 1, val: res[i][j - 1]))
                }
                res[i][j] = neighbours.min()?.val ?? res[i][j]
                res[i][j] = res[i][j] != Int.max ? res[i][j] + 1 : res[i][j]
            }
        }
        for i in (0..<res.count).reversed() {
            for j in (0..<res[i].count).reversed() {
                if res[i][j] <= 1 {
                    continue
                }
                var neighbours = [MatrixNode(x: i, y: j, val: res[i][j])]
                if i < res.count - 1 {
                    neighbours.append(MatrixNode(x: i + 1, y: j, val: res[i + 1][j]))
                }
                if i > 0 {
                    neighbours.append(MatrixNode(x: i - 1, y: j, val: res[i - 1][j]))
                }
                if j < res[i].count - 1 {
                    neighbours.append(MatrixNode(x: i, y: j + 1, val: res[i][j + 1]))
                }
                if j > 0 {
                    neighbours.append(MatrixNode(x: i, y: j - 1, val: res[i][j - 1]))
                }
                res[i][j] = neighbours.min()?.val ?? res[i][j]
                res[i][j] = res[i][j] != Int.max ? res[i][j] + 1 : res[i][j]            }
        }
        
        return res
    }
    
    func orangesRotting(_ grid: [[Int]]) -> Int {
        var res = grid
        
        res = grid.map { row in
            row.map { num in
                num == 0 ? -1 : num == 2 ? 0 : Int.max
            }
        }
        print(res)
        for _ in 1...3 {
            for i in 0..<res.count {
                for j in 0..<res[i].count {
                    if res[i][j] <= 1 {
                        continue
                    }
                    var neighbours = [MatrixNode(x: i, y: j, val: res[i][j])]
                    if i < res.count - 1 && res[i + 1][j] >= 0 {
                        neighbours.append(MatrixNode(x: i + 1, y: j, val: res[i + 1][j]))
                    }
                    if i > 0 && res[i - 1][j] >= 0 {
                        neighbours.append(MatrixNode(x: i - 1, y: j, val: res[i - 1][j]))
                    }
                    if j < res[i].count - 1 && res[i][j + 1] >= 0 {
                        neighbours.append(MatrixNode(x: i, y: j + 1, val: res[i][j + 1]))
                    }
                    if j > 0 && res[i][j - 1] >= 0 {
                        neighbours.append(MatrixNode(x: i, y: j - 1, val: res[i][j - 1]))
                    }
                    res[i][j] = neighbours.min()?.val ?? res[i][j]
                    res[i][j] = res[i][j] != Int.max ? res[i][j] + 1 : res[i][j]
                }
            }
            for i in (0..<res.count).reversed() {
                for j in (0..<res[i].count).reversed() {
                    if res[i][j] <= 1 {
                        continue
                    }
                    var neighbours = [MatrixNode(x: i, y: j, val: res[i][j])]
                    if i < res.count - 1 && res[i + 1][j] >= 0 {
                        neighbours.append(MatrixNode(x: i + 1, y: j, val: res[i + 1][j]))
                    }
                    if i > 0 && res[i - 1][j] >= 0 {
                        neighbours.append(MatrixNode(x: i - 1, y: j, val: res[i - 1][j]))
                    }
                    if j < res[i].count - 1 && res[i][j + 1] >= 0 {
                        neighbours.append(MatrixNode(x: i, y: j + 1, val: res[i][j + 1]))
                    }
                    if j > 0 && res[i][j - 1] >= 0 {
                        neighbours.append(MatrixNode(x: i, y: j - 1, val: res[i][j - 1]))
                    }
                    res[i][j] = neighbours.min()?.val ?? res[i][j]
                    res[i][j] = res[i][j] != Int.max ? res[i][j] + 1 : res[i][j]            }
            }
        }
        var result = 0
        
        for row in res {
            for num in row {
                result = max(num, result)
            }
            print(row)
        }
        return result == Int.max ? -1 : result
    }
    
    func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
        var result: ListNode?
        var pointer: ListNode?
        var list1 = list1, list2 = list2
        if list1 == nil {
            return list2
        }
        if list2 == nil {
            return list1
        }

        while list1 != nil && list2 != nil {
            if result == nil {
                result = ListNode(min(list1!.val, list2!.val))
                pointer = result
                if result?.val == list1!.val {
                    list1 = list1!.next
                } else {
                    list2 = list2!.next
                }
            } else {
                pointer!.next = ListNode(min(list1!.val, list2!.val))
                pointer = pointer!.next
                if pointer!.val == list1!.val {
                    list1 = list1!.next
                } else {
                    list2 = list2!.next
                }
            }
        }
        
        while list1 != nil {
            if result == nil {
                result = ListNode(list1!.val)
                pointer = result
            } else {
                pointer!.next = ListNode(list1!.val)
                pointer = pointer!.next
            }
            list1 = list1!.next
        }
        
        while list2 != nil {
            if result == nil {
                result = ListNode(list2!.val)
                pointer = result
            } else {
                pointer!.next = ListNode(list2!.val)
                pointer = pointer!.next
            }
            list2 = list2!.next
        }

        return result
    }
    
    func reverseList(_ head: ListNode?) -> ListNode? {
        var result: ListNode?
        var pointer = result
        let stack = Stack<Int>()
        var head = head
        while head != nil {
            stack.push(head!.val)
            head = head?.next
        }
        while !stack.isEmpty() {
            if result == nil {
                result = ListNode(stack.pop())
                pointer = result
            } else {
                pointer!.next = ListNode(stack.pop())
                pointer = pointer!.next
                
            }
        }
        return result
    }
    
    func combine(_ n: Int, _ k: Int) -> [[Int]] {
        let nums = Array(1...n)
        var res: [[Int]] = []
        if k == 1 {
            return nums.map { num in
                [num]
            }
        }
        res = combine(nums, k)
        return res
    }
    
    private func combine(_ nums: [Int], _ k: Int) -> [[Int]] {
        if k == 1 {
            return nums.map { n in
                [n]
            }
        }
        var res: [[Int]] = []
        var nums = nums
        for _ in 1...nums.count - k + 1 {
            let number = nums.remove(at: 0)
            var temp = combine(nums, k - 1)
            temp = temp.map { row in
                var res = row
                res.insert(number, at: 0)
                return res
            }
            res.append(contentsOf: temp)
        }
        
        return res
    }
    
    func permute(_ nums: [Int]) -> [[Int]] {
        if nums.count == 1 {
            return [nums]
        }
        if nums.count == 2 {
            return [nums, nums.reversed()]
        }
        var res: [[Int]] = []
        
        res = heapPermutation(nums, nums.count, nums.count)
        
        return res
    }
    
    private func heapPermutation(_ nums: [Int], _ n: Int, _ size: Int) -> [[Int]] {
        var res: [[Int]] = []
        var nums = nums
        if size == 1 {
            var temp: [Int] = []
            for i in 0..<n {
                temp.append(nums[i])
            }
            return [temp]
        }
        
        for i in 0..<size {
            res.append(contentsOf: heapPermutation(nums, n, size - 1))
            print(size, i, nums)
            if size % 2 == 1 {
                swap(&nums, 0, size - 1)
            } else {
                swap(&nums, i, size - 1)
            }
        }
        return res
    }
    
    private func swap(_ nums: inout [Int], _ a: Int, _ b: Int) {
        let t = nums[a]
        nums[a] = nums[b]
        nums[b] = t
    }
    
    func letterCasePermutation(_ s: String) -> [String] {
        var s = s.uppercased()
        var letterCount = 0
        var dict: [Int: DefaultIndices<String>.Element] = [:]
        
        for i in s.indices {
            if s[i].isLetter {
                dict[letterCount] = i
                letterCount += 1
            }
        }
        var res = [String]()
        res.append(s)

        if letterCount == 0 {
            return res
        }
                
        for i in 1..<twoPower(letterCount) {
            var pattern = String(i, radix: 2)
            while pattern.count < letterCount {
                pattern.insert("0", at: pattern.startIndex)
            }
            
            for j in 0..<pattern.count {
                if pattern[pattern.index(pattern.startIndex, offsetBy: j)] == "0" {
                    let c = s.remove(at: dict[j]!)
                    s.insert(Character(c.uppercased()), at: dict[j]!)
                    
                } else {
                    let c = s.remove(at: dict[j]!)
                    s.insert(Character(c.lowercased()), at: dict[j]!)
                }
            }
            res.append(s)
        }
        return res
    }
    
    private func twoPower(_ n: Int) -> Int {
        if n == 0 {
            return 1
        }
        if n == 1 {
            return 2
        }
        var res = 2
        for _ in 1..<n {
            res *= 2
        }
        return res
    }
    
    func climbStairs(_ n: Int) -> Int {
        var map: [Int: Int] = [:]
        return climbStairs(n, &map)
    }
    
    private func climbStairs(_ n: Int, _ map: inout [Int: Int]) -> Int {
        var count = 0
        if n == 1 {
            map[1] = 1
            return 1
        }
        if n == 2 {
            map[2] = 2
            return 2
        }
        if let saved = map[n] {
            return saved
        }
        count += climbStairs(n - 1, &map)
        count += climbStairs(n - 2, &map)
        map[n] = count
        return count
    }
    
    func rob(_ nums: [Int]) -> Int {
        var memo = [Int](repeating: -1, count: nums.count)
        return rob(nums, nums.count - 1, &memo)
    }
    
    private func rob(_ nums: [Int], _ i: Int, _ memo: inout [Int]) -> Int {
        if i < 0 {
            return 0
        }
        if memo[i] >= 0 {
            return memo[i]
        }
        let res = max(rob(nums, i - 2, &memo) + nums[i], rob(nums, i - 1, &memo))
        memo[i] = res
        return res
    }
    
    func minimumTotal(_ triangle: [[Int]]) -> Int {
        var resultTriangle = triangle
        minimumTotalRec(triangle, &resultTriangle, 0)
        return resultTriangle[0][0]
    }
    
    private func minimumTotalRec(_ triangle: [[Int]], _ resultTriangle: inout [[Int]], _ index: Int) {
        if index == triangle.count - 1 {
            return
        }
        minimumTotalRec(triangle, &resultTriangle, index + 1)
        for i in 0..<triangle[index].count {
            resultTriangle[index][i] += min(resultTriangle[index + 1][i], resultTriangle[index + 1][i + 1])
        }
    }
    
    func isPowerOfTwo(_ n: Int) -> Bool {
        n&(n-1) == 0 && n > 0
    }
    
    func hammingWeight(_ n: Int) -> Int {
        var n = n
        var count = 0
        while n > 0 {
            n = n&(n - 1)
            count += 1
        }
        return count
    }
    
    func reverseBits(_ n: Int) -> Int {
        var str = ""
        var n = n
        var res = 0
        for _ in 0...31 {
            str.append(String(n&1))
            n = n >> 1
        }
        for i in 0..<str.count {
            res += (Int(String(str[str.index(str.startIndex, offsetBy: i)])) ?? 1) * 1<<(31 - i)
        }
        return res
    }
    
    func singleNumber(_ nums: [Int]) -> Int {
        var xorRes = nums[0]
        for i in 1..<nums.count {
            xorRes = xorRes ^ nums[i]
        }
        xorRes = nums.reduce(0, { partialResult, n in
            n^partialResult
        })
        return xorRes
    }
}
