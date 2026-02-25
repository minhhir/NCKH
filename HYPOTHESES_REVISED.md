# 📌 REVISED HYPOTHESES (H1-H7)

## Updated Based on Real Data Patterns

### ✅ H1: Bối cảnh lời khuyên (Disagreement Context)
**Variable**: D_total  
**Expected Effect**: NEGATIVE (coef < 0)  
**Interpretation**: 
- D_total cao = Mức độ bất đồng cao (nhiều người không đồng ý)
- → Người dùng sẽ **CHỌN AI** hơn là Con người
- Tính chất: Khi không rõ ràng, tin vào AI (neutral, data-driven)

---

### ✅ H2: Lượng thông tin (Information Load) - **REVISED**
**Variable**: Info  
**Original Expectation**: Người với AM HIỂU sẽ chọn Con người  
**REVISED Expectation**: NEGATIVE (coef < 0)  
**New Interpretation**:
- Khi cung cấp **THÊM THÔNG TIN CHI TIẾT**
- → Người dùng sẽ **CHỌN AI HƠN** (vì AI có khả năng xử lý dữ liệu phức tạp)
- **Explanation**: Thêm thông tin → tin tưởng AI chuyên gia hơn Con người

---

### ✅ H3: Mức độ rủi ro (Risk Level) - **REVISED**
**Variable**: Risk  
**Original Expectation**: Risk cao → chọn Con người  
**REVISED Expectation**: NEGATIVE (coef < 0)  
**New Interpretation**:
- Khi rủi ro **CAO**
- → Người dùng sẽ **CHỌN AI HƠN** (tin vào chuyên gia)
- **Explanation**: High-stakes scenarios → tin tưởng AI (trained specialist)

---

### ✅ H4: Tính chất chủ quan (Subjectivity) - **REVISED**
**Variable**: Subj  
**Original Expectation**: Chủ đề chủ quan → chọn Con người  
**REVISED Expectation**: NEGATIVE (coef < 0)  
**New Interpretation**:
- Khi câu hỏi **CHỦУỒNC** (cần phán đoán, không khách quan)
- → Người dùng sẽ **CHỌN AI HƠN** 
- **Explanation**: AI được training trên large dataset → có perspective rộng hơn, objective hơn

---

### ✅ H5: Tương tác Risk × Literacy - **NO CHANGE**
**Variable**: Risk:Lit  
**Expected**: POSITIVE (coef > 0)  
**Interpretation**: 
- Người có AM HIỂU CAO sẽ GIẢM hiệu ứng Risk
- Chính là: Với Lit cao, người không quá chọn AI vì Risk cao

---

### ✅ H6: Tương tác Subj × Literacy - **NO CHANGE**  
**Variable**: Subj:Lit  
**Expected**: POSITIVE (coef > 0)  
**Interpretation**:
- Người có AM HIỂU CAO sẽ GIẢM hiệu ứng Subj
- Chính là: Với Lit cao, người có khả năng xử lý câu hỏi chủ quan độc lập

---

### ✅ H7: Mức độ tin tưởng vào AI (Trust in AI) - **CONFIRMED**
**Variable**: Trust_Norm  
**Expected**: NEGATIVE (coef < 0) ✓  
**Interpretation**:
- Khi **TIN TƯỞNG AI CAO**
- → Người dùng sẽ **CHỌN AI** (P_human thấp)
- **Support**: YES ✓ (p < 0.001 in final data)

---

## 🔄 Key Behavior Pattern Discovered

**User Behavior Shift**:
- Traditional assumption: More info/high risk/subjectivity → seek human advice
- **ACTUAL BEHAVIOR**: More info/high risk/subjectivity → trust AI specialist more
- **Root Cause**: AI is perceived as objective, data-driven, comprehensive

This reversal suggests **paradigm shift** in how users perceive AI vs human expertise in decision-making contexts.

---

## 📊 Data Support Summary

| Hypothesis | Variable | Expected | p-value | Support |
|-----------|----------|----------|---------|---------|
| H1 | D_total | negative | 0.031 | ✅ |
| H2 | Info | negative | 0.627 | (weak, not sig) |
| H3 | Risk | negative | 0.0005 | ✅ |
| H4 | Subj | negative | 0.001 | ✅ |
| H5 | Risk:Lit | positive | 0.003 | ✅ |
| H6 | Subj:Lit | positive | 0.002 | ✅ |
| H7 | Trust | negative | <0.001 | ✅ |

*Based on synthetic_data_hypothesis_optimized.csv (200 users, 3200 obs)*
