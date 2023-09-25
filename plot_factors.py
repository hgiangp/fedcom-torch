from parse_log import * 
import matplotlib.pyplot as plt 

file_name = './logs/server_model.log'
lipschitz = parse_Lipschitz_factor(file_name)
convex = parse_convex_factor(file_name)

plt.figure()
plt.plot(lipschitz, '--', label='Lipschitz')
plt.plot(convex, ':', label='strongly convex')
plt.grid()
plt.legend()
plt.savefig('lischitz.png')
plt.show()
plt.close()

# plt.figure()
# # plt.plot(lipschitz, '--', label='Lipschitz')

# plt.grid()
# plt.legend()
# plt.savefig('convex.png')
# plt.show()
# plt.close()