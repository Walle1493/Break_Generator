# Calculate the edit distance of two sentences

def minDistance(s1, s2):
    word1 = s1.strip().split()
    word2 = s2.strip().split()
    dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
    for i in range(len(word1)+1):
        dp[i][0] = i
    for j in range(len(word2)+1):
        dp[0][j] = j
    for i in range(1, len(word1)+1):
        for j in range(1, len(word2)+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
    return dp[-1][-1]


if __name__ == "__main__":
    s1 = "company >> distributed the film"
    s2 = "Who distributed the film UHF?"
    result = minDistance(s1, s2)
    print(result)
