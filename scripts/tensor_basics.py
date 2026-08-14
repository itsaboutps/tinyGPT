import torch


print("=" * 60)
print("1. SCALAR")
print("=" * 60)

scalar = torch.tensor(5.0)

print("Tensor:")
print(scalar)

print("Shape:")
print(scalar.shape)

print("Dimensions:")
print(scalar.ndim)


print()
print("=" * 60)
print("2. VECTOR")
print("=" * 60)

vector = torch.tensor(
    [10.0, 20.0, 30.0]
)

print("Tensor:")
print(vector)

print("Shape:")
print(vector.shape)

print("Dimensions:")
print(vector.ndim)


print()
print("=" * 60)
print("3. MATRIX")
print("=" * 60)

matrix = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

print("Tensor:")
print(matrix)

print("Shape:")
print(matrix.shape)

print("Dimensions:")
print(matrix.ndim)
# --------------------------------------------
print()
print("=" * 60)
print("4. GPT STYLE TENSOR")
print("=" * 60)

B = 2 #boxes //  mean number of matrix inside means we process 2 sentences together.
T = 4 # number of rows in each matrix //means each sentence has 4 tokens
C = 3 #number of elements in each matrix //means each token is represented by 3 numbers (embedding size 3).
# 👉 2 batches
# 👉 each batch has 4 tokens
# 👉 each token has 3 features/numbers
x = torch.randn(B, T, C)

print("Tensor:")
print(x)

print()

print("Shape:")
print(x.shape)

print()

print("Dimensions:")
print(x.ndim)


print()
print("=" * 60)
print("5. TENSOR EXAMPLE - 1")
print("=" * 60)
# ==========================================================
x = torch.tensor(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],

        [
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
        ],
    ]
)
print(x[0])
print(x[0, 1])
print(x[0, 1, 2])



# x.shape = [32, 128, 512]
# x[5, 20]
# give me the representation of:

# token position 20

# from training sequence 5

# and the result contains:
#     512 numbers
print()
print("=" * 60)
print("6. TENSOR EXAMPLE 2")
print("=" * 60)

B = 2

T = 4

C = 3


x = torch.randn(B, T, C)

print(x.shape)


print()
print("=" * 60)
print("7. TENSOR EXAMPLE SLICING :::::::::::::::3")
print("=" * 60)

x = torch.tensor(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],

        [
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
        ],
    ]
)
print("x[:, 0]")
print(x[:, 0])
print("x[:, 0, :]")
print(x[:, 0, :])
print("x[:, :2, :]")
print(x[:, :2, :])

print()
print("=" * 60)
print("8. TENSOR EXAMPLE RESHAPE :::::::::::::::3")
print("=" * 60)
x = torch.arange(12)

print(x)
print(x.shape)
# y = x.reshape(3, 4)
print(x.reshape(3, 4))
print(x.reshape(2, 6))
print(x.reshape(3, -1))
# print(x.reshape(5, 3)) RuntimeError: shape '[5, 3]' is invalid for input of size 12

print()
print("=" * 60)
print("9. TENSOR EXAMPLE TRANSPOSE :::::::::::::::3")
print("=" * 60)
x = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)
print(x.transpose(0, 1)) #We swapped dimensions:dimension 0 ↔ dimension 1

print()
print("=" * 60)
print("10. TENSOR EXAMPLE Why attention needs transpose :::::::::::::::3")
print("=" * 60)
print()
print("=" * 60)
print("11. TENSOR EXAMPLE Matrix multiplication :::::::::::::::3")
print("=" * 60)

A = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0]
    ]
)

B = torch.tensor(
    [
        [5.0, 6.0],
        [7.0, 8.0]
    ]
)
C = A @ B
print(C)

print()
print("=" * 60)
print("12. TENSOR EXAMPLE Broadcasting:::::::::::::::3")
print("=" * 60)
x = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

bias = torch.tensor(
    [10, 20, 30]
)
print(x+ bias)


print()
print("=" * 60)
print("13. TENSOR EXAMPLE unsqueeze:::::::::::::::3")
print("=" * 60)

x = torch.tensor(
    [1, 2, 3]
)
print(x.unsqueeze(0))
print(x.unsqueeze(1))


print()
print("=" * 60)
print("13. TENSOR EXAMPLE squeeze Opposite of unsqueeze:::::::::::::::3")
print("=" * 60)
print(x.squeeze(0))






print()
print("=" * 60)
print("5. GPT SHAPE EXPERIMENT")
print("=" * 60)


B = 2
T = 4
C = 6

x = torch.randn(B, T, C)

print("Input x:")
print(x.shape)


# Imagine these are query and key matrices.
q = x
k = x


print()
print("Query shape:")
print(q.shape)

print("Key shape:")
print(k.shape)


k_transposed = k.transpose(-2, -1)

print()
print("Transposed key shape:")
print(k_transposed.shape)


attention_scores = q @ k_transposed

print()
print("Attention scores shape:")
print(attention_scores.shape)


attention_weights = torch.softmax(
    attention_scores,
    dim=-1
)

print()
print("Attention weights shape:")
print(attention_weights.shape)


print()
print("Attention weights:")
print(attention_weights)


print()
print("Sum of each attention row:")
print(
    attention_weights.sum(dim=-1)
)
