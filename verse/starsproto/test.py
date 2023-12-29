from starset import StarSet

test = StarSet(1,[2],3)
test.show()

new = test.superposition(2, [4])
new.show()

new = new.superposition(2, [5,6])
