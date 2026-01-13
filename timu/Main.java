package timu;

import java.util.HashSet;
import java.util.Set;

public class Main {
    public static void main(String[] args) {
        Solution solution = new Solution();
        String[] testCases = { "abcabcbb", "bbbbb", "", "au" };
        for (int i = 0; i < testCases.length; i++) {
            String s = testCases[i];
            int result = solution.lengthOfLongsetSubstring(s);
            System.out.println(result);
        }
    }

    static class Solution {
        public int lengthOfLongsetSubstring(String s) {
            Set<Character> window = new HashSet<>();
            int n = s.length();
            int left = 0;
            int maxLength = 0;
            for (int right = 0; right < n; right++) {
                char c = s.charAt(right);
                while (window.contains(c)) {
                    window.remove(s.charAt(left));
                    left++;
                }
                window.add(c);
                maxLength = Math.max(maxLength, right - left + 1);
            }
            return maxLength;
        }
    }
}
