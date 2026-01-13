class Solution {
    public boolean canJump(int[] nums) {
        int flag[] = new int[10001];
        flag[nums.length-1]=1;
        for(int i=nums.length-2;i>=0;i--){
            for(int j=1;i<=nums[i];j++)
            {
                if(flag[i+j]==1)
                {
                    flag[i]=1;
                    break;
                }
            }
        }
        if(nums[0]==1)
        {
            return true;
        }
        else{
            return false;
        }
    }
}