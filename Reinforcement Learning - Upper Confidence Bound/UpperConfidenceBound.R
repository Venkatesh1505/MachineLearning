
#read data
df = read.csv('Ads_CTR_Optimisation.csv')

#random selection
N = 10000
d = 10
ads_selected = integer(0)
tot_reward = 0
for (n in 1:N) {
  ad = sample(1:d,1)
  ads_selected = append(ads_selected, ad)
  tot_reward = tot_reward + df[n,ad]
}

#Upper Confidence Bound Algorithm
N = 10000
d= 10
total_reward = 0
sum_of_rewards = integer(d)
number_of_selection = integer(d)
ads_selected = integer(0)
for (n in 1:N) {
  ad = 0
  max_upper_bound = 0
  for (i in 1:d) {
    if (number_of_selection[i] > 0) {
      avg_reward = sum_of_rewards[i] / number_of_selection[i]
      delta = sqrt(3/2 * log(n)/number_of_selection[i])
      upper_bound = avg_reward + delta
    }
    else{
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound){
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected,ad)
  number_of_selection[ad] = number_of_selection[ad] + 1
  reward = df[n,ad]
  sum_of_rewards[ad] = sum_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

#Visualisation
hist(ads_selected, col = 'blue',
     main = paste('Upper confidence bound - Number of times user clicks on each ad'),
     xlab = 'Ads',
     ylab = 'Number of times clicked')