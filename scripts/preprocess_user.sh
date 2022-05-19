USER_IDS=( 2f647ee5db504196adb5d1425a9f737b d47cded7ec8b4a31a0c3796752e5fde1 ff55306707044d7389f8a75002fb6824 70939c954e944b11834e38782bad518b 3375299d28ab42378e622ad3e878024c bb4b007c9d4c4ad88600d7d230bf2c88 3c05e83ec8704dd8a72f3110365bf32e 1d952a4522784c959468e768e6827af8 a13ad5ad7f094f368400e2627761d998 14428b1dbb1143908b24474b282016f5 2bbf9b68ec784293a2bc4286bfe6285d f10deb915c9540d980eaec61a122e8a3 0fff49fe0bd543f8a6c483dee43ca963 c88f413c9fdf486e824e6841084fa727 a5688ac3113242bfae93fa9dbc973708 ad0de5a20c5a4ff0b1f76ffd122b72a4 aa7a8d565a8b45b7b1cd6f17cce42027 b31325622775467293b226316011bba3 dea888412b154dffb055de4a5427ab4d 07bcc42fda3b436b85e6789dc6661579 11dfdccd56a64465a4f9fb86438dcebb a51f5d25ce6f44bf9a4cb1cb5d9bb1d3 204bcfb7d5ce4406bcc2eb918a07fbd2 107d1f48cc874db3be5c2b109b8e7f74 46b2d49376db49eb94b1a648b7883eb0 b0646ae36bf34d4081994d4bfee1d21c 137861cee6ce450185a02936239d7354 66a757dafb314039b3113119a7198e9d 7bc7d9d88cf3475aa5a5b6abf4e0f4ee 104f86ddd0ac4355badec0ea6e71c141 45c6239b47394c4b9240a0f423b4ad96 e2dc5381a30f41f78cdc6ed523918256 e15a024306be4a9580554f50463872e0 eb1b6474d0384cc4be7b7aaea8a4da91 01b6e4951b1d4913b2d4f2a4e22cec8a 1b33cb2e0ce548ce9ce3f48685984488 6ceb8d591d89477f9fec1dbe02b74f6f 90ab13dfd3714632bec0e8c0f7c0d756 9fc37565c86e4dca802644cfc85bc567 5ec488987d7f473492c25d79af6f365d 47317d553db3431e9fe128afa5b849d9 1717708c7af5468188e808a8d09b5c98 5cdb7f8238d64073827e0e1f46851a64 dd948e1de9fa4327a603edd7c120e09c 3db737df635a497ca8e185db47cce07c 47823a67c3b04caaa3730345c80b02f9 44103fcb65dd4a35b6de0acedf4dded7 3d10f55326ce454199f1d1a2ea4e8453 f657620fa4b94be89dfc300b94203233 70540704017c4d0d9a0f2eb3c855ec89 3d5ecc7659474f39a4549bdf4a9e4484 3055bd8e3be647a9a7bee490085d2040 0477057b436143b38fec19e2cbe0c8d5 e410957a70ea4366a813728d5ef9d5cb e7227a703fd248eb9db4ce290a2a8e70 f337460b0db0474aa4981b3f41784297 31620e4203e44d9d9e7f326cf069a6af 29cdecd26a474c9a867bc34f39eb828a 612bf6acd7204c60883d6f99b1ae20b3 29a0389c19584728876a41ac136549ac cd3ec1c6832043a39c5b74012419b67c fb611fb606e540a0848063fe9e3972c7 f510b8bd8b824efa8706a59d484c13fd 753480244b6b486c96b8a66ca41659b1 0b78040e2cef4ac39c8d106ab0cdee3f efbc23a792744bddbc17d6843276632a f7f943886c4a4ad999a9148b3ce94caa ed780ce41340451496923751e069d0bd d1d798f536004a0c827dc5229e10a31f f1efcf0e7f2345a2baff62cf4f597297 ab9b615fffd2424c8b3d8ae25a0d0d0d 1fa00f010a9c4a53a7b6b8b6940f7696 776646d756a04492a7e7fdb3c813f918 a42030e8be3446419ef6a71004e736d3 024a1b78e88a4b62ae4c7f7ba36456f3 ce7b4487b3dc4e42bacc707d34276913 3586941216ac4e2284ca3428a13517d8 b15611f064144daeb3cb1a3ceb868adf 85a55d1be60d4f5a8c20ebe7b34977e9 c84639e82a334aaebb4131defa2378dd 1941b4c7e63747eb83a4a44135c06e75 9354baf4c6ef4015a103c23ea197929a 7fd9e07983a1423aa9792ab68d9b614d 9c29c123bf8a4e2da6cd61889c33a761 074ed05e84c74e438a7417794d58879f 2fa8c4a0b8d945a1a32ca683a30f8430 4612734ce9d54721901241e33d9adea2 )
for USER_ID in "${USER_IDS[@]}"
do
    echo $USER_ID
    python ./preprocessing/split_data_user.py --data=data/gld_${1}_vision_tensors.pkl --train=data/users/gld_${1}_${USER_ID}_vision_train.pkl --test=data/users/gld_${1}_${USER_ID}_vision_test.pkl --users $USER_ID 1> data/users/${USER_ID}.txt
    python ./preprocessing/generate_negatives.py --data_file=data/users/gld_${1}_${USER_ID}_vision_train.pkl --out_file=data/users/gld_${1}_${USER_ID}_vision_train_pos_neg.pkl
    python ./preprocessing/generate_negatives.py --data_file=data/users/gld_${1}_${USER_ID}_vision_test.pkl --out_file=data/users/gld_${1}_${USER_ID}_vision_test_pos_neg.pkl
done