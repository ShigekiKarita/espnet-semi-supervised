#!/bin/sh
dev_cer=$(grep -e Avg -e SPKR -m 2 $1/decode_test_dev*/result.txt  | awk '{print $11}' | tail -n 1)
eval_cer=$(grep -e Avg -e SPKR -m 2 $1/decode_test_eval*/result.txt  | awk '{print $11}' | tail -n 1)

dev_wer=$(grep -e Avg -e SPKR -m 2 $1/decode_test_dev*/result.wrd.txt  | awk '{print $11}' | tail -n 1)
eval_wer=$(grep -e Avg -e SPKR -m 2 $1/decode_test_eval*/result.wrd.txt  | awk '{print $11}' | tail -n 1)
echo WER dev $dev_wer % eval $eval_wer %

dev_ser=$(grep -e Avg -e SPKR -m 2 $1/decode_test_dev*/result.wrd.txt  | awk '{print $12}' | tail -n 1)
eval_ser=$(grep -e Avg -e SPKR -m 2 $1/decode_test_eval*/result.wrd.txt  | awk '{print $12}' | tail -n 1)
echo SER dev $dev_ser % eval $eval_ser %


echo "| dev-CER  | eval-CER  | dev-WER | eval-WER | dev-SER | eval-SER | path "
echo "| $dev_cer | $eval_cer | $dev_wer | $eval_wer | $dev_ser | $eval_ser | $1 "
