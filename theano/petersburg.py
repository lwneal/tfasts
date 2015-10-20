import random

def play():
  pot = 2.0
  while random.choice(['heads', 'tails']) is 'heads':
    pot *= 2.0
  return pot

def play_n_times(n):
  return sum([play() for i in range(n)]) / n

if __name__ == '__main__':
  print "A man in St. Petersburg offers you a bet. The pot starts at $2"
  print "A coin is flipped. If it is tails, you win the $2 pot."
  print "If it is heads, then the pot doubles, and you flip again."
  print "Three heads in a row and you win $8."
  print "Six heads in a row and you win $64. Ten and you win $1024"
  print
  print "How many dollars would you pay to play this game?"
  print "$ ",
  cost = raw_input()

  print "Playing once, you win: {0}".format(play())
  print "100 plays: average winnings {0}".format(play_n_times(100))
  print "1000 plays: average winnings {0}".format(play_n_times(1000))
  print "10000 plays: average winnings {0}".format(play_n_times(10000))
  print "100000 plays: average winnings {0}".format(play_n_times(100000))
  print "1000000 plays: average winnings {0}".format(play_n_times(1000000))

  print "You paid {0}. Was it worth it?".format(cost)
