from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import json
import math
from tqdm import tqdm
with open ("/home/user/legal_llm/client_secret.json", "r") as f:
    cred = json.load(f)
cred

PROJECT_ID = cred['installed']['project_id']
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Define where your source files are stored
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="/home/user/legal_llm/application_default_credentials.json"
paths = [
 "https://drive.google.com/file/d/11BHQ9FnqaQlSukXAASiGRp_v7obOY8SW/view?usp=sharing",
 "https://drive.google.com/file/d/12JPkhR8GyoyvGRyzQElabKbp3xWak7sL/view?usp=sharing",
 "https://drive.google.com/file/d/13Bc1F2q5yx7R9CBJtChcm9S6zEbMJ1G3/view?usp=sharing",
 "https://drive.google.com/file/d/148lybRG1Jvii4IaQJctA0gCZLIcjKdTB/view?usp=sharing",
 "https://drive.google.com/file/d/18D1gTeNWzAbDLSYEVSlgXv6Fj6ku33U6/view?usp=sharing",
 "https://drive.google.com/file/d/1A7B5nxXhtrY3TVCAGJhaLegB3xk2fUaj/view?usp=sharing",
 "https://drive.google.com/file/d/1ANtuAKvc3lpcNaJaIoHibh2GF57kE1rj/view?usp=sharing",
 "https://drive.google.com/file/d/1AiIt-sa7cUNr6HGxYJ_MUUy67dbg3D9l/view?usp=sharing",
 "https://drive.google.com/file/d/1B8QeRG2FwqK7qoyAgCzFzCmH3NzQBDQf/view?usp=sharing",
 "https://drive.google.com/file/d/1BiqKFPmqSKWGEuPcbDWtUwQCsvEQD3-n/view?usp=sharing",
 "https://drive.google.com/file/d/1DX2RTM5NTQ2FAX7rBAlJT5BRlP4j_tie/view?usp=sharing",
 "https://drive.google.com/file/d/1EJ3w_ew0zwj3wVNKL5k3A46XCEfZSXZu/view?usp=sharing",
 "https://drive.google.com/file/d/1KZe2P6kXyT3XkcFtnCAsrEok-1CeEN8H/view?usp=sharing",
 "https://drive.google.com/file/d/1N6KgS9NEb32CECGlIhQwci_zhK89JV1o/view?usp=sharing",
 "https://drive.google.com/file/d/1Nt7jaEzQBcRraIS6Ip8yvpIa8FL_BAZg/view?usp=sharing",
 "https://drive.google.com/file/d/1SF1V1QrpqF9MIgIbgnkJgEi06RKtOta2/view?usp=sharing",
 "https://drive.google.com/file/d/1TarFTyCu1ll2bwwprOXMRCiFumiPXg9a/view?usp=sharing",
 "https://drive.google.com/file/d/1XfUyBO4v0cthfDk_FdGwaSWFLJ-re-F1/view?usp=sharing",
 "https://drive.google.com/file/d/1a5kwz2ksiUGwCl_R-PtcdLf9IV3SiUbX/view?usp=sharing",
 "https://drive.google.com/file/d/1b70WVOFLU8oZPhz0Hc4r7-bG-VjaEQdn/view?usp=sharing",
 "https://drive.google.com/file/d/1bR5UeX7jXGR7jSzsSB0JBfV9v3Y8Q8u0/view?usp=sharing",
 "https://drive.google.com/file/d/1iAvt1f-QmK5qvYHxmqgtnY3-YvLZg5XG/view?usp=sharing",
 "https://drive.google.com/file/d/1kW0_XZxc38vYw2A4TVe48PNZqCXET0Ox/view?usp=sharing",
 "https://drive.google.com/file/d/1lhVqo1lkDKlLgGnGoNpY_GXQi1p3mpqg/view?usp=sharing",
 "https://drive.google.com/file/d/1o4fm0CCGRqypGN1rZaoylAyM0URBhToe/view?usp=sharing",
 "https://drive.google.com/file/d/1oRr8jGO-_1Libnvj7Gt6-zJlaUVtVXpe/view?usp=sharing",
 "https://drive.google.com/file/d/1plmRsapTYRHFsxqKSGSFJVWIegDwQknr/view?usp=sharing",
 "https://drive.google.com/file/d/1qNMEWWeJr_t3aS3is8E_YYrCIeGbOAqa/view?usp=sharing",
 "https://drive.google.com/file/d/1tul9Bg8ov9lVYQhpgVRyhRGf44TplkKI/view?usp=sharing",
 "https://drive.google.com/file/d/1uoInYOSAz3SopqOl6QeBeRQpD5O6apWs/view?usp=sharing",
 "https://drive.google.com/file/d/1v16iCsP4BGx5wNvm9OlC0x7QRptnzGZH/view?usp=sharing",
 "https://drive.google.com/file/d/1wSh6mS2zmDlJG1qzMDTvwd-jJ9Auj_W2/view?usp=sharing",
 "https://drive.google.com/file/d/1yBoSLDsmWO_YiW3dnt13OsjKJOJdF9bt/view?usp=sharing",
 "https://drive.google.com/file/d/1zD5MYT_LltTGObLVKiaxyJDPqcIo88s_/view?usp=sharing",
 "https://drive.google.com/file/d/1-DIDGjiunPnOAx9CIbI3nltR05wvuV6b/view?usp=sharing",
 "https://drive.google.com/file/d/1-WDJFEMR9IShQbt9QaP4yotKjK6s9lUW/view?usp=sharing",
 "https://drive.google.com/file/d/19jbnNrC8z3YdN5okf-Y7zFGACrP8HJ68/view?usp=sharing",
 "https://drive.google.com/file/d/1GVboRha4Y57Y85FZVy7MMEhu8O9rgURs/view?usp=sharing",
 "https://drive.google.com/file/d/1LAqSWrP2BXLiXybJWE8iq8NgDS0oJjTp/view?usp=sharing",
 "https://drive.google.com/file/d/1PVghk-jKUbzpfh5NUHUaKr_zVejP2uD1/view?usp=sharing",
 "https://drive.google.com/file/d/1WlH5FGJeLTjc0ipJReOIVlJNCbQdRST8/view?usp=sharing",
 "https://drive.google.com/file/d/1tghZhK8Tyiuu1tX2Mebxnk2fC56PAwhW/view?usp=sharing",
 "https://drive.google.com/file/d/1uKBRCyPT6UmRogP-_7GI81Z97_IJ1YZ1/view?usp=sharing",
 "https://drive.google.com/file/d/1-JOs-PyKzJkbTzX4Kr_XLkPC5vzVhXUm/view?usp=sharing",
 "https://drive.google.com/file/d/1-We9AdJeLbEVAf_Ww01gX34GMT3PjuNI/view?usp=sharing",
 "https://drive.google.com/file/d/1-pa221muGHxTdI9BYYclp6hiHD3iDqVk/view?usp=sharing",
 "https://drive.google.com/file/d/11TbsUlVM0-OvMMZf8HIGPAB1cJhvrDzM/view?usp=sharing",
 "https://drive.google.com/file/d/11gJTChZ9GURvKZPS3ySv6YS6_35BcqLO/view?usp=sharing",
 "https://drive.google.com/file/d/12l076Wwhhd47H2WNTLlUpnt5F2i_AufU/view?usp=sharing",
 "https://drive.google.com/file/d/13GJbyHKyWufB6v9UDKJMkuuMhWLhBvKZ/view?usp=sharing",
 "https://drive.google.com/file/d/13dA0qxMBXVeJGdYa1ve8PMkTk-AK0T-0/view?usp=sharing",
 "https://drive.google.com/file/d/13kTHpwIU75Jp1U46tcr1xZlmi8tV3bLb/view?usp=sharing",
 "https://drive.google.com/file/d/14ULPDcfz8j8beNwNh---CeaAPfGLYr52/view?usp=sharing",
 "https://drive.google.com/file/d/15IhV4NVqSKhhNmzAWCRNzN29E3p9Wp28/view?usp=sharing",
 "https://drive.google.com/file/d/1CKMasTlqKxd4MVoxk47x2IfgCTSrFHvJ/view?usp=sharing",
 "https://drive.google.com/file/d/1EMTTkXhSK-VzRrj46wme6RUsIUBjVOLw/view?usp=sharing",
 "https://drive.google.com/file/d/1EXUYje2PeaBL76zOCXqhdIbeQwQ69Kb5/view?usp=sharing",
 "https://drive.google.com/file/d/1FfGK11uCmRTZSQghx6zF7WcQR-ghKFju/view?usp=sharing",
 "https://drive.google.com/file/d/1G2lcnDOsaBomEZ-up8T-Q14L8lVwWXAL/view?usp=sharing",
 "https://drive.google.com/file/d/1GCjIM8omrsokCD1NqXWhu0Pn_NtLBIOT/view?usp=sharing",
 "https://drive.google.com/file/d/1Gzcust_QVCxZkLLqSEFE6255kbKljB44/view?usp=sharing",
 "https://drive.google.com/file/d/1Hebm6fmcHv-SiKm29h1kMlQJ3HMPRKbi/view?usp=sharing",
 "https://drive.google.com/file/d/1J2q1Uh2TKJx00dA9EWwIlJSBfNgVK3dE/view?usp=sharing",
 "https://drive.google.com/file/d/1JHUfdSExQIg4Ac3m5YCQtWO1crItEqyR/view?usp=sharing",
 "https://drive.google.com/file/d/1Jlgk82RR2kyk3s1Lv-Ip7qFq6qf44Wm6/view?usp=sharing",
 "https://drive.google.com/file/d/1JnnIG39PwYR8eb5Hmsx_pVguktOZBvuN/view?usp=sharing",
 "https://drive.google.com/file/d/1MmtczVR79R-mCAcVa7OIPntIb6XGywjR/view?usp=sharing",
 "https://drive.google.com/file/d/1PaFY0SP4oZzCQAGRrHpxkC0aqiuDRMPS/view?usp=sharing",
 "https://drive.google.com/file/d/1UAJ-Z0dJZZwMIgv6pLEs37RK9vnBRSYB/view?usp=sharing",
 "https://drive.google.com/file/d/1UGL5W7638rLIMfRvHWY1pa__dF-WUzry/view?usp=sharing",
 "https://drive.google.com/file/d/1VPji6xlcjgqrwnigUYzet-e6uUoM-mbZ/view?usp=sharing",
 "https://drive.google.com/file/d/1WaIMo0No_9w18Y6vGFIVESqzqtNmApB2/view?usp=sharing",
 "https://drive.google.com/file/d/1WmStkDu4cK0IF-8gd-ZY9p0uMy92gICy/view?usp=sharing",
 "https://drive.google.com/file/d/1XWJ1ZTGgcn2g2j_2luEXtyAl3rpCm_iM/view?usp=sharing",
 "https://drive.google.com/file/d/1XZK-ZELmAT11aDqzy0Cx6dh9ESvi6_ad/view?usp=sharing",
 "https://drive.google.com/file/d/1a0_s-pt4xNwjRHIoVcaJCcyHOP3a4-JC/view?usp=sharing",
 "https://drive.google.com/file/d/1adQ4gf8pEFDPR4tuf-MQL04nCg0aPJYC/view?usp=sharing",
 "https://drive.google.com/file/d/1czP2QJ2C-dRqc-KCemPTf0KLum3ze5Zg/view?usp=sharing",
 "https://drive.google.com/file/d/1eVTGRKAP0hU-k_vhB2R_x-lYxbiwzoMK/view?usp=sharing",
 "https://drive.google.com/file/d/1eXhD7jFF-ELjO5BjaKPbotTjDm8AniNs/view?usp=sharing",
 "https://drive.google.com/file/d/1gDnWB_N8RJnKnUGiR-u3zb-gr4QFQQiL/view?usp=sharing",
 "https://drive.google.com/file/d/1h9Hps7uSGnU7PnHCFPmX6OY67gBq7MB6/view?usp=sharing",
 "https://drive.google.com/file/d/1hcnD3ldlm8qEmAiGJGNwyG-bn82zbjtb/view?usp=sharing",
 "https://drive.google.com/file/d/1i5vieMPzjEpqi8fgHiHGJm3ktOGrvvy8/view?usp=sharing",
 "https://drive.google.com/file/d/1o8uWmiSnMEpckPilWmlw2rvVZasWiv3N/view?usp=sharing",
 "https://drive.google.com/file/d/1p56OljLyNYYfm4ZxNZrmLrqD6Tdv87nf/view?usp=sharing",
 "https://drive.google.com/file/d/1pnsCAZ6RQeakK7-NSr1tt_KeH0_crmhX/view?usp=sharing",
 "https://drive.google.com/file/d/1sGAegJc33RQzw7fAYiJ4ZY00vGTehDE-/view?usp=sharing",
 "https://drive.google.com/file/d/1uywpM3aHmY4sx4qNCIIhtWAzNVF23QcK/view?usp=sharing",
 "https://drive.google.com/file/d/1voaf3tNWI2tB5H8-gssdpqXRGexIGzOW/view?usp=sharing",
 "https://drive.google.com/file/d/1xYhQs2m0yZorcPmagZI2PLhkV8H8y9G1/view?usp=sharing",
 "https://drive.google.com/file/d/1xtfy9zKH_-t8-uwY8tXtFNfKVIsZjDlv/view?usp=sharing",
 "https://drive.google.com/file/d/1ycQHqH8cQBX_puydQT98LYy0O5UZ3s8F/view?usp=sharing",
 "https://drive.google.com/file/d/10DR1bZJuykjNOjcB4nG-r12E6fPn-ALx/view?usp=sharing",
 "https://drive.google.com/file/d/10H91e--v4OG_9Tf_nFd6SKsqRTqk1XW2/view?usp=sharing",
 "https://drive.google.com/file/d/13n_-V7G6hocztNVo2w_VomdWaGdszo8a/view?usp=sharing",
 "https://drive.google.com/file/d/19S1Ufp7TU63LPvFAb8CNDou-nD9suhsA/view?usp=sharing",
 "https://drive.google.com/file/d/1P-4635miy4I9trMwi-mysEfagmaPbrfT/view?usp=sharing",
 "https://drive.google.com/file/d/1Rh4ponMksGoL9wZw_pLAv30CX-bD8WL3/view?usp=sharing",
 "https://drive.google.com/file/d/1ZQGPXRKdayOE3Eu7GoA89pt79h6VtZTz/view?usp=sharing",
 "https://drive.google.com/file/d/1fKz8--AVz4-3AEsm9CWhx8pR2q81QfBO/view?usp=sharing", 
 "https://drive.google.com/file/d/14K16pwra4MKhHIUKHpwz9siJo8YB0oGZ/view?usp=sharing",
 "https://drive.google.com/file/d/1Yx5ggwOeJcUOPWhBuSHqVJbI2bQ7drw2/view?usp=sharing",
 "https://drive.google.com/file/d/1sn9h1laKdqIaOMeKHwEzJnahGMk8zTOx/view?usp=sharing",
 "https://drive.google.com/file/d/1yIa5jrpN1pwJQ9n1QiN564ZLjlO6vwKH/view?usp=sharing", 
 "https://drive.google.com/file/d/1-P8enqI4IXqiVJVmm5kqnpzgXQ15BjVS/view?usp=sharing",
 "https://drive.google.com/file/d/1-qeF7xcmzbGyBXpjmzBbJMnvIBssHK9M/view?usp=sharing",
 "https://drive.google.com/file/d/10H_kdE8UuIT1laiMK1tQq5T1di49Hoj8/view?usp=sharing",
 "https://drive.google.com/file/d/10Q7Zh8rPilz_yOruuY_cQc5p6wPqCnC-/view?usp=sharing",
 "https://drive.google.com/file/d/10Swl8APfICrWpguEC_znhsBxq0H9Kzzd/view?usp=sharing",
 "https://drive.google.com/file/d/116VqFns1_w2U2KzJz789_PFM5guAzik6/view?usp=sharing",
 "https://drive.google.com/file/d/11iTsEsjF5knaVkQM30_zFnvWUi4Sr-iZ/view?usp=sharing",
 "https://drive.google.com/file/d/120n5N0VTC28Qw2ISGwuhgi_xj14lIk-c/view?usp=sharing",
 "https://drive.google.com/file/d/12dOvsdVYKgDtkGGkS_2QNEQ2dcLpElyG/view?usp=sharing",
 "https://drive.google.com/file/d/134WM1RbiOKM3102hCdUs7R4bnfXY50JZ/view?usp=sharing",
 "https://drive.google.com/file/d/13UhsPaNbGAMU5Z2C2COjxe_jLVBogtoh/view?usp=sharing",
 "https://drive.google.com/file/d/141eiEID8Fm3TJGJ5oYxFnOG8hgoHZ-Ax/view?usp=sharing",
 "https://drive.google.com/file/d/17RpALi-pXYkY0JrBt5-7gyBL_4E5Iqo5/view?usp=sharing",
 "https://drive.google.com/file/d/17x-uM0lWAUn0vN4aydGgAfsJheBRh4_V/view?usp=sharing",
 "https://drive.google.com/file/d/182QWiOxBiTj7slu1h8nanWHu29W7tnsv/view?usp=sharing",
 "https://drive.google.com/file/d/189zWOEK8fayr4trVMjSPWtOWvfV71Mje/view?usp=sharing",
 "https://drive.google.com/file/d/18VXoh68N79_8xEl3Du8jWtWcsSKdNL1m/view?usp=sharing",
 "https://drive.google.com/file/d/18Waai1MZx4bqhYU36lrop10XtUwDVQQj/view?usp=sharing",
 "https://drive.google.com/file/d/194Yl3f5IZ4BRypHPgCQbUmMesN5--l1k/view?usp=sharing",
 "https://drive.google.com/file/d/19tMdiApEZU1qHEVcuLkQm67BtYaVgkkN/view?usp=sharing",
 "https://drive.google.com/file/d/1A9v4YPkrPBIXjIOA8eD0X2y8IsKDQrMG/view?usp=sharing",
 "https://drive.google.com/file/d/1AP5JBnlv2YjzlaRcmo1PRxfvQT6GpEvT/view?usp=sharing",
 "https://drive.google.com/file/d/1B9ucbJ-609tCR-uKxqfjKGWcT23piTjp/view?usp=sharing",
 "https://drive.google.com/file/d/1BIghGztqVdQdz2wwttWWVAeVqMJY_L9o/view?usp=sharing",
 "https://drive.google.com/file/d/1BaVKjqEdN7Gmyn-ORzmzQu1JbtJwoE-j/view?usp=sharing",
 "https://drive.google.com/file/d/1CP8q_iaBF9jiwIXDoDu6IK2jPuhVdpkg/view?usp=sharing",
 "https://drive.google.com/file/d/1DQTdUnrpfJJyhtaHzdBi2yQq6_PeG3-E/view?usp=sharing",
 "https://drive.google.com/file/d/1Ds8JJxTO5wLELC0n2o8ht0Xx6otfy8rR/view?usp=sharing",
 "https://drive.google.com/file/d/1DsEiv6UMppKnr6Xxw49Btl4jr4YUbd3L/view?usp=sharing",
 "https://drive.google.com/file/d/1EI01DlS_JQyarrk2D5Lo9vRbIjzL_rPr/view?usp=sharing",
 "https://drive.google.com/file/d/1F9y0YM16Y92ZbWgy_9L9jyDxBnLZ327D/view?usp=sharing",
 "https://drive.google.com/file/d/1FveIX-roDymlLaQfeo4lM2UNH16ihd6k/view?usp=sharing",
 "https://drive.google.com/file/d/1GIVgl-enqTZ6yB49bVLrnLNr0EsqSeL9/view?usp=sharing",
 "https://drive.google.com/file/d/1GQVYxywKKov7QdFX0Hh5og2aGuN_zmLw/view?usp=sharing",
 "https://drive.google.com/file/d/1GeEe68reFeJNa0_Ap1VpiPXLF-Smxv6_/view?usp=sharing",
 "https://drive.google.com/file/d/1GlodEUhATZTj16uU1T1Y_m-ofhYKnmHC/view?usp=sharing",
 "https://drive.google.com/file/d/1HIWb7DOLNAfZhxlr2yn3GNMkyLF3IyS5/view?usp=sharing",
 "https://drive.google.com/file/d/1HvLaRd5iPpjo2DqmNRGgKUtIRkNR4rxm/view?usp=sharing",
 "https://drive.google.com/file/d/1IJhdhocWdRUDepQzthrVyEV37UCYGyLv/view?usp=sharing",
 "https://drive.google.com/file/d/1IkJSaBFP9TbBylyvUX9d5J0FSLKXcvYH/view?usp=sharing",
 "https://drive.google.com/file/d/1JIJ030R6VkmPqCY96XKcwX0a4VZeZJN_/view?usp=sharing",
 "https://drive.google.com/file/d/1JILuQ-lVXkyUnhLhJvUqeQv2Ufllq7Jq/view?usp=sharing",
 "https://drive.google.com/file/d/1JMsZHHSmzZvbuCv_sfQ6CJ2S_JcGuM9e/view?usp=sharing",
 "https://drive.google.com/file/d/1JpdgWOAiWTBe13ScluYT6ekjVwfJ8Suw/view?usp=sharing",
 "https://drive.google.com/file/d/1KJ2-qipxWeWswdIsPoG7fmITVHhpauEA/view?usp=sharing",
 "https://drive.google.com/file/d/1KLx643UEQNZNCXgARn1CMfRGCFrJKMM2/view?usp=sharing",
 "https://drive.google.com/file/d/1KqX4PqK6Fqx029_r-iHyFkYg4QQ3Fipm/view?usp=sharing",
 "https://drive.google.com/file/d/1LgTZndbftDchaAVoPp_77cSm_BI-d65F/view?usp=sharing",
 "https://drive.google.com/file/d/1Lu5UjlJcl4bMxbq9Y9HYwAcp4jynK_qk/view?usp=sharing",
 "https://drive.google.com/file/d/1M1NxC2sAbNRtMlPcs33FbnZCcf7PQEx1/view?usp=sharing",
 "https://drive.google.com/file/d/1Mizhs96zrgwU1o43hltd73DHFG9L5wBK/view?usp=sharing",
 "https://drive.google.com/file/d/1Mvsfdq6RXtW6sRfr8w-_FrihWuA-Wcaw/view?usp=sharing",
 "https://drive.google.com/file/d/1NH5iF4opS_2G8555wuXJuWawDMcSu17r/view?usp=sharing",
 "https://drive.google.com/file/d/1Nc5Aj0leHoBWup4crG087xouLzyQPIIk/view?usp=sharing",
 "https://drive.google.com/file/d/1NeTsWzFDUC_Q1IT8aZL8XpuEdT8ScM-s/view?usp=sharing",
 "https://drive.google.com/file/d/1NwFxswuT8eS8D6ykYWxRZfqxpAndUfcO/view?usp=sharing",
 "https://drive.google.com/file/d/1OF3G1PiE5z9RnTJVMf28W7dhNIijjfVi/view?usp=sharing",
 "https://drive.google.com/file/d/1OIeg6hmPCTfB4nLuVVDhGiP_T6QDWsEV/view?usp=sharing",
 "https://drive.google.com/file/d/1PBpPk2jz5x9JFtuShlE4gYJ_I_YXgsyC/view?usp=sharing",
 "https://drive.google.com/file/d/1Ppa1qozow0m61kwma4GAYq16hn7At_uk/view?usp=sharing",
 "https://drive.google.com/file/d/1Q9SWkFnG4OvQ1mAnyW6E1ImYPy1t6_ew/view?usp=sharing",
 "https://drive.google.com/file/d/1R9SHuAcUXqz2OKxieOIyKPUS-0AAisNO/view?usp=sharing",
 "https://drive.google.com/file/d/1RcMBRrSvZS1SU0VvLOokjnTIKqFMY9WR/view?usp=sharing",
 "https://drive.google.com/file/d/1RmImHwcX7Wnx_6AONSZG8JcMgNMz9VZk/view?usp=sharing",
 "https://drive.google.com/file/d/1Rq3uH4Q9zokO7yE9xcPukPMhMGmp4235/view?usp=sharing",
 "https://drive.google.com/file/d/1RrfY__HElS-q9iY3Q90z9iXT2qnNeQmY/view?usp=sharing",
 "https://drive.google.com/file/d/1S03mc79ciO07ri-09zfw3EdyCTRGuYo9/view?usp=sharing",
 "https://drive.google.com/file/d/1S49GjpOkjr3IU7ieZT-A2ByWEFa8rBaE/view?usp=sharing",
 "https://drive.google.com/file/d/1SHiK-Aah1M1KSnHC-nKqWbSe_hE_EomM/view?usp=sharing",
 "https://drive.google.com/file/d/1SiC7VuJFhp67XDfrE3SdG4AnqklH3GL1/view?usp=sharing",
 "https://drive.google.com/file/d/1T4g4QRuS2Rh3hCZTFT9XuZkyPzbklKq0/view?usp=sharing",
 "https://drive.google.com/file/d/1UPiZn_sPIZdVeQcI2I_ls7FPBNYXz2r4/view?usp=sharing",
 "https://drive.google.com/file/d/1VN8O0wnRVwQb4YHmHMytvorE9wUsh0WG/view?usp=sharing",
 "https://drive.google.com/file/d/1VZ1oYSjJ3sJ-_GBSIgaVNX6rhY_spagx/view?usp=sharing",
 "https://drive.google.com/file/d/1VfQSPKG9AculORn_J-GLdxGxtGPpwvjH/view?usp=sharing",
 "https://drive.google.com/file/d/1WSbZXC8ph-KhMX4oZWoR7zIse7ZXBmVv/view?usp=sharing",
 "https://drive.google.com/file/d/1Wm22zr8wfdYyehOzh0v_WgecNcFqkiqY/view?usp=sharing",
 "https://drive.google.com/file/d/1X9zSiBu16Z_UsjJiRt1eDY9PlKrpVOeH/view?usp=sharing",
 "https://drive.google.com/file/d/1XOUbsDfPxzr0GjsErXUEBX_N9VU41rkr/view?usp=sharing",
 "https://drive.google.com/file/d/1XjnfKh0lTyBo2qJDXiAgZy0KxPlvqQu0/view?usp=sharing",
 "https://drive.google.com/file/d/1YDIkE-55ulOH8ofQNpmT_dUVJCK6rsL8/view?usp=sharing",
 "https://drive.google.com/file/d/1Y_vdj37aOezlxWorN7NOUOidv1vVmbtX/view?usp=sharing",
 "https://drive.google.com/file/d/1YvwI8RnAEP8buazfnG5Ysbmt5x1PrgAj/view?usp=sharing",
 "https://drive.google.com/file/d/1ZR0y79DU8qgHD64E0hNmkhYkuOvLixjp/view?usp=sharing",
 "https://drive.google.com/file/d/1Zf9sb8_DHOFy250QMpPAy1sGnaHF13Q4/view?usp=sharing",
 "https://drive.google.com/file/d/1_G3PicqYFqqsqJOMPp4mH7P0jOELeosY/view?usp=sharing",
 "https://drive.google.com/file/d/1_Kfzg2C6QZbD_3hhRUpcrUPSCHRlH4Im/view?usp=sharing",
 "https://drive.google.com/file/d/1_dxt6LY0B2yWNkdwsdFh8nVskZpqqu4J/view?usp=sharing",
 "https://drive.google.com/file/d/1_iMCurnFn3Wd6OMGg5mkxAJGZpDfb-cF/view?usp=sharing",
 "https://drive.google.com/file/d/1_oH0HJnOr-yL7l_puVYrJk_UWS2A8ZFy/view?usp=sharing",
 "https://drive.google.com/file/d/1_zLtW9LoUJgrQcpmm8XHVMjdnIpl5ajM/view?usp=sharing",
 "https://drive.google.com/file/d/1aL24bF2dsTL20x7m6jt9yVuoo630pWtU/view?usp=sharing",
 "https://drive.google.com/file/d/1bQN1NRLI8gzAcAzQOvIr7amtX2qHJVA1/view?usp=sharing",
 "https://drive.google.com/file/d/1bbN_jySZkarBf8k3QF-P_qZdFNNGIRbQ/view?usp=sharing",
 "https://drive.google.com/file/d/1bpnhp6Ub1UoVMWhiegWEw0VeQxqmGSbo/view?usp=sharing",
 "https://drive.google.com/file/d/1c3l2CmwIr1qfH-htiCL_xEZjPep8rqpC/view?usp=sharing",
 "https://drive.google.com/file/d/1cZKeMhvvZvRbBIfBMUcnIXDOjEWLJVPw/view?usp=sharing",
 "https://drive.google.com/file/d/1dZwbCBA-RYE_EFuNYyBhhLUk9rTHWiCU/view?usp=sharing",
 "https://drive.google.com/file/d/1eT4IsEAwNl0Bv2tdaJTBMPPPgfvvptgX/view?usp=sharing",
 "https://drive.google.com/file/d/1eioO-BY1DuYLFiRgDnTuLKfUCMsO9T0I/view?usp=sharing",
 "https://drive.google.com/file/d/1fqQ_0zFdnDxurwlDacdz8ESCglzgb7Iu/view?usp=sharing",
 "https://drive.google.com/file/d/1gi7ca2WnBI7MTB2KRx8kMph78Z_vjm5W/view?usp=sharing",
 "https://drive.google.com/file/d/1gsI95k14iI-JvEl4lzMDkzY-CPFnCPom/view?usp=sharing",
 "https://drive.google.com/file/d/1hD3lXoxnmQ58rWBwgBRHuV3TsNrV6UkW/view?usp=sharing",
 "https://drive.google.com/file/d/1hHVvNpbz9YeeBac7vg8xtrXppVO8nReV/view?usp=sharing",
 "https://drive.google.com/file/d/1hw8YLipd4uTA5YjLt0Mi6P8pQ-bhHNGN/view?usp=sharing",
 "https://drive.google.com/file/d/1irsagBDHDl6fP2AjjmRmN5Udl428cVWh/view?usp=sharing",
 "https://drive.google.com/file/d/1jtZIAhc6EZvrs3vNn3hstpN_azsANG5s/view?usp=sharing",
 "https://drive.google.com/file/d/1juGkCLAFJ8Fn6-7VDSZN7gErX-kHkjNG/view?usp=sharing",
 "https://drive.google.com/file/d/1kUHtDpguZD_I7YnIpBc_ASUCegxL7-5b/view?usp=sharing",
 "https://drive.google.com/file/d/1kdURBjmgwCaHjIwA07jbvOg9uhsmIazC/view?usp=sharing",
 "https://drive.google.com/file/d/1kwUmXFL5Mhor8je4wWHUwK-q-KP7joST/view?usp=sharing",
 "https://drive.google.com/file/d/1lIs0LzTHOsfbOOT4ksoaiCnvMqhNqeZy/view?usp=sharing",
 "https://drive.google.com/file/d/1lXObDUWT1_8qyQ3YXmBDgDzPq6hpofMb/view?usp=sharing",
 "https://drive.google.com/file/d/1ldA53MBtqbw-FxUia8_IDJQMRbaukFzS/view?usp=sharing",
 "https://drive.google.com/file/d/1lvBamnBXabVOR0Ntn8SUVzB37zCSuBcY/view?usp=sharing",
 "https://drive.google.com/file/d/1m5LUo2Hgts5IayKiYwCDPVIeHvxw5Pjq/view?usp=sharing",
 "https://drive.google.com/file/d/1mBdYIb3cqiMUCdW9PT8sPvZ6LsWNX4yd/view?usp=sharing",
 "https://drive.google.com/file/d/1mTB2aBetRE3uRAHLEB0pFEjzMjL4ZzAP/view?usp=sharing",
 "https://drive.google.com/file/d/1mfZd0ICGHi4olmx0hPFgWHsq_zuzlLWx/view?usp=sharing",
 "https://drive.google.com/file/d/1mlFIVKM9n_wW9eJkA-bzr8wasAYxW4gk/view?usp=sharing",
 "https://drive.google.com/file/d/1mvR-A6yhvXWgvNQrF5NoStPzt1Te7D3s/view?usp=sharing",
 "https://drive.google.com/file/d/1my7lRq6TP_QHPHW93fIuZgDm3M46MpID/view?usp=sharing",
 "https://drive.google.com/file/d/1n70hZ6ACRQ-YZNZmBQy9AxZUqBhySNAh/view?usp=sharing",
 "https://drive.google.com/file/d/1nNw1O4n6OnDELvF0Q4G80EECXNuRDg64/view?usp=sharing",
 "https://drive.google.com/file/d/1nXdFMIg1zPOjMjDVkKzj7qSCivfLjZ6c/view?usp=sharing",
 "https://drive.google.com/file/d/1onn2Ycef2rjXvSLH1SOKqKvFdUlbbDLb/view?usp=sharing",
 "https://drive.google.com/file/d/1pFLgMxM28aHFNhA-3_gyfx_-thKH3Aed/view?usp=sharing",
 "https://drive.google.com/file/d/1pH-rIM3q4hhLAha_OhCxq_A9fWV-8lgH/view?usp=sharing",
 "https://drive.google.com/file/d/1pKKU_qVLUAz23cCJ1iMWltqq_raom0hK/view?usp=sharing",
 "https://drive.google.com/file/d/1pccBIivicZwUZ-lzvY2Slkl7AFwXllJ3/view?usp=sharing",
 "https://drive.google.com/file/d/1phiDcaycndEvmDBHXaJZa-raFjWA61do/view?usp=sharing",
 "https://drive.google.com/file/d/1pzQEI_ckcSVceCZGuGd3WVHpwFlXOX9O/view?usp=sharing",
 "https://drive.google.com/file/d/1q2fXKUSp-RtITc3RWXTtyrDb83Fgq6rx/view?usp=sharing",
 "https://drive.google.com/file/d/1qgVj5KmgJT03b2be4IgTsR37tma48Ffo/view?usp=sharing",
 "https://drive.google.com/file/d/1rCjKxDaiDtdGentEKSbYTO4SOETFD1v8/view?usp=sharing",
 "https://drive.google.com/file/d/1rHuRPVHPcxAf-X58dFwdkFPhOpYqPI6J/view?usp=sharing",
 "https://drive.google.com/file/d/1ra9Kko4H8LBQmjNC_57TfNFDlOyWX0Sx/view?usp=sharing",
 "https://drive.google.com/file/d/1rslmTkuO4BxhbkAYyL2yRu6v3I-ZdCYK/view?usp=sharing",
 "https://drive.google.com/file/d/1sbOXUrLxSvepJWc6q6hRPb22PLKI0W08/view?usp=sharing",
 "https://drive.google.com/file/d/1tSg5l-Iu5M9IWf0JlGUcZP8D4T4gs0H1/view?usp=sharing",
 "https://drive.google.com/file/d/1tmP2kc-uEIKeWnAkXaNGzHFJw91oj92T/view?usp=sharing",
 "https://drive.google.com/file/d/1u1XfUVBRonFgJy6157CniXRL-S2VOjFl/view?usp=sharing",
 "https://drive.google.com/file/d/1urJboa1wtdXUd-4yUPxtoJlmQicWxucP/view?usp=sharing",
 "https://drive.google.com/file/d/1vUcw86jN10JMixdDjLpnccP-hW8Y5He0/view?usp=sharing",
 "https://drive.google.com/file/d/1vUzrcHdB3LgFgc7WgfojbMtN4YtgxTR2/view?usp=sharing",
 "https://drive.google.com/file/d/1wABsyXuvf8hIxQbuwCefn9k8UoJqdo-0/view?usp=sharing",
 "https://drive.google.com/file/d/1x7HZuQhLHKAsBvnNOvnug9kQ_bDEBLxh/view?usp=sharing",
 "https://drive.google.com/file/d/1xW7QdBDiu3TbWUDzNjJt4RZrRYChwbvU/view?usp=sharing",
 "https://drive.google.com/file/d/1yizSL8imLqQv1lTc9LBqVmTkDeH57ong/view?usp=sharing",
 "https://drive.google.com/file/d/1yo946yYOuJHZb594xUtTRsSviOShl66_/view?usp=sharing",
 "https://drive.google.com/file/d/1zAF0d2bEJ_uQXo3uzFn3OE3bpzuaU2kc/view?usp=sharing",
 "https://drive.google.com/file/d/1zaMdbzzXyx2gj8r4XwQiMjOTIp2ULpsE/view?usp=sharing",
 "https://drive.google.com/file/d/1zbQdUbtGnMT09oHThBl7LNQUPi6P0txj/view?usp=sharing",
 "https://drive.google.com/file/d/1zbp80iIFiZO19zKg-7r00Fnr6QPp8Arj/view?usp=sharing",
 "https://drive.google.com/file/d/1EqRkV5nFou6GTH0K2ztNlJpHG_X3d5lW/view?usp=sharing",
 "https://drive.google.com/file/d/1H3_jQP9_qA2RJzST_1PDL4lISdvj6Q6l/view?usp=sharing",
 "https://drive.google.com/file/d/1Km9omjbmrWsiRDITtxxieY1jweg2H1cV/view?usp=sharing",
 "https://drive.google.com/file/d/1UT7KEMLDBwe9K80b4fHSrJS27js_ecX5/view?usp=sharing",
 "https://drive.google.com/file/d/1jk0iFbe2KlSZZusQp72ZhIDXeH2CoJY9/view?usp=sharing",
 "https://drive.google.com/file/d/1lK1DUQvDDiU-HUO3Q7_kmVZcLPANNIUd/view?usp=sharing",
 "https://drive.google.com/file/d/1mb7g29q-SOQoaY2GMNYSujehysWZOy0g/view?usp=sharing",
 "https://drive.google.com/file/d/1pES9VRD42WUF7R1F72y0GGA0M-QVfZZJ/view?usp=sharing",
 "https://drive.google.com/file/d/1wf4SKcnZCbtyjg5oXnHdjsBB9tm1poD7/view?usp=sharing",
 "https://drive.google.com/file/d/1z4MN-hSh74YKkmHl_0-VSmqZ3FEo2Wiu/view?usp=sharing",
 "https://drive.google.com/file/d/1RS-leOX0Kj9iqKboJECuEnulXNM5_-UJ/view?usp=sharing",
 "https://drive.google.com/file/d/1kqBeJkNPFYEXf1ub7tCOJDJq3X5BwUG5/view?usp=sharing",
 "https://drive.google.com/file/d/1rfvRWt7hQNFzk7QBXJ2e-GYrqZGo0hBr/view?usp=sharing",
 "https://drive.google.com/file/d/1zX6orMobdwyFhkbbDaajc8W73U5OaoXc/view?usp=sharing",
 "https://drive.google.com/file/d/1CYoxu6u-emfaz3axD8ntZH-qF23zHfUq/view?usp=sharing",
 "https://drive.google.com/file/d/1J8Byagkxv9z8JEHzVx1UEgCCSKz4cSjV/view?usp=sharing",
 "https://drive.google.com/file/d/1O6OhR9TP1xBHmH_Ly24QNK66DlA1jbGi/view?usp=sharing",
 "https://drive.google.com/file/d/1PHURLTvALWrmhRb2c9EV4U2w4EYhQW8a/view?usp=sharing",
 "https://drive.google.com/file/d/1Qkja2rNHd-edPVl507L8PDZyCbb9nRjj/view?usp=sharing",
 "https://drive.google.com/file/d/1StpUgt3xL9B9l0vXDxl4rUiLkwspkn1s/view?usp=sharing",
 "https://drive.google.com/file/d/1VlcXsJiOnSJ_ToL26oenalywBrV6Q5Yn/view?usp=sharing",
 "https://drive.google.com/file/d/1aaN2gPPHuuyQcXNeNpHNY1xMKlBsLymE/view?usp=sharing",
 "https://drive.google.com/file/d/1gWXyoAKAAZFHiZbPjJg-VNQuowpR-CuY/view?usp=sharing",
 "https://drive.google.com/file/d/1qERxaz3S2Hy_mXIWCJuxCZEUTPNQXaOB/view?usp=sharing",
 "https://drive.google.com/file/d/1tG6HTzHp2uab_9sYI60fIGPh8eeq1RFi/view?usp=sharing",
 "https://drive.google.com/file/d/11f_7TUYcX7pKX1F-PHyTsmeRFmhqPLBI/view?usp=sharing",
 "https://drive.google.com/file/d/1TpQv_1ntUH7BuaxRMb-muSJ0ouid4ahv/view?usp=sharing",
 "https://drive.google.com/file/d/1f12KOQRiQHELbv4kvhRP2NYT_CrMrmEd/view?usp=sharing",
 "https://drive.google.com/file/d/1Go22p8M6ZyPRQJbPiRsBq37DfMTKYUOv/view?usp=sharing",
 "https://drive.google.com/file/d/1faTrzYVecK_ReVQ4SZaq-s7pWAZKbXWO/view?usp=sharing"
]

# Optional: change display_name and embedding model if needed
# display_name = "dkncxk"

# embedding_model_config = rag.RagEmbeddingModelConfig(
#     vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
#         publisher_model="publishers/google/models/text-embedding-005"
#     )
# )

# # Create the corpus for your RAG project
# rag_corpus = rag.create_corpus(
#     display_name=display_name,
#     backend_config=rag.RagVectorDbConfig(
#         rag_embedding_model_config=embedding_model_config
#     )
# )

# paths1 = paths
# # Import files into the corpus, configure chunking as needed
# for path in tqdm(paths1, desc="Importing files"):
#     rag.import_files(
#         rag_corpus.name,
#         [path],  # Wrap in list to keep parameter type expected
#         transformation_config=rag.TransformationConfig(
#             chunking_config=rag.ChunkingConfig(
#                 chunk_size=512,
#                 chunk_overlap=100,
#             ),
#         ),
#         max_embedding_requests_per_min=1000
#     )

# # Set up retrieval configuration
# rag_retrieval_config = rag.RagRetrievalConfig(
#     top_k=3,
#     filter=rag.Filter(vector_distance_threshold=0.5)
# )

# # Retrieve relevant context for a specific query
# response = rag.retrieval_query(
#     rag_resources=[
#         rag.RagResource(rag_corpus=rag_corpus.name),
#     ],
#     text="What is RAG and why is it helpful?",
#     rag_retrieval_config=rag_retrieval_config,
# )

# print(response)

# # Use Gemini model for full generation
# rag_retrieval_tool = Tool.from_retrieval(
#     retrieval=rag.Retrieval(
#         source=rag.VertexRagStore(
#             rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
#             rag_retrieval_config=rag_retrieval_config,
#         ),
#     ),
# )

# rag_model = GenerativeModel(
#     model_name="gemini-2.0-flash-001",
#     tools=[rag_retrieval_tool]
# )

# response = rag_model.generate_content( "help in establishing a restaurant business in Abu Dhabi?")
# print(response.text)


display_name = "my_corpus_8"

embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

# Create the corpus for your RAG project
rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_model_config
    )
)

# Helper: Split a list into chunks of size n
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Process files in batches of 25
batch_size = 25
paths1 = paths # your full list of file paths

for batch in tqdm(chunk_list(paths1, batch_size), total=math.ceil(len(paths1)/batch_size), desc="Importing files in batches"):
    rag.import_files(
        rag_corpus.name,
        batch,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=512,
                chunk_overlap=100,
            ),
        ),
        max_embedding_requests_per_min=1000
    )

# Set up retrieval configuration
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=3,
    filter=rag.Filter(vector_distance_threshold=0.5)
)

# Retrieve relevant context for a specific query
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(rag_corpus=rag_corpus.name),
    ],
    text="What is RAG and why is it helpful?",
    rag_retrieval_config=rag_retrieval_config,
)

print(response)

# Use Gemini model for full generation
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    ),
)

rag_model = GenerativeModel(
    model_name="gemini-2.0-flash-001",
    tools=[rag_retrieval_tool]
)

response = rag_model.generate_content( "help in establishing a restaurant business in Abu Dhabi?")
print(response.text)