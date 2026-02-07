#!/usr/bin/env python3
"""
Data Contracts Validation Script
==================================
Validates YAML data contracts for MLOps Assignment 2

Author: Sahil Kamboj (B23CS1061)
Course: ML-DL-Ops (CSL7120)
"""

import yaml
import re
import sys
from pathlib import Path
from datetime import datetime

def validate_yaml_syntax(file_path):
    """Validate YAML syntax"""
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)

def validate_odcs_structure(contract):
    """Validate ODCS required sections"""
    required_sections = ['dataContractSpecification', 'info', 'schema', 'quality']
    missing = [s for s in required_sections if s not in contract]
    return len(missing) == 0, missing

def validate_scenario_1_rides(contract):
    """Validate Scenario 1: Ride-Share requirements"""
    checks = []
    
    # Check logical field mappings
    schema = contract.get('schema', {})
    properties = schema.get('properties', {})
    required_fields = ['ride_id', 'pickup_timestamp', 'passenger_id', 
                      'driver_rating', 'fare_amount', 'distance_meters']
    
    fields_present = all(field in properties for field in required_fields)
    checks.append(("Rides: All logical field mappings present", fields_present))
    
    # Check PII tagging
    pii_tagged = properties.get('passenger_id', {}).get('pii') == True
    checks.append(("Rides: PII tagging on passenger_id", pii_tagged))
    
    # Check SLA
    sla = contract.get('sla', {})
    freshness_ok = sla.get('freshness', {}).get('threshold') == '30 minutes'
    checks.append(("Rides: SLA freshness = 30 minutes", freshness_ok))
    
    # Check quality rules
    quality = contract.get('quality', [])
    fare_rule = any('fare_amount' in r.get('expression', '') and '>= 0' in r.get('expression', '') 
                   for r in quality)
    rating_rule = any('driver_rating' in r.get('expression', '') 
                     for r in quality)
    distance_rule = any('distance' in r.get('name', '').lower() or
                       'distance_meters' in r.get('column', '')
                       for r in quality)
    
    checks.append(("Rides: Quality rule 'fare_amount_non_negative' found", fare_rule))
    checks.append(("Rides: Quality rule 'driver_rating_range' found", rating_rule))
    checks.append(("Rides: Quality rule 'distance_not_null' found", distance_rule))
    
    return checks

def validate_scenario_2_orders(contract):
    """Validate Scenario 2: E-commerce requirements"""
    checks = []
    
    schema = contract.get('schema', {})
    properties = schema.get('properties', {})
    
    # Check order_total has minimum: 0
    order_total = properties.get('order_total', {})
    min_zero = order_total.get('minimum') == 0
    checks.append(("Orders: order_total has minimum: 0", min_zero))
    
    # Check status enum mapping
    status = properties.get('status', {})
    has_enum = 'enum' in status
    checks.append(("Orders: Status enum correctly mapped", has_enum))
    
    # Check quality rules
    quality = contract.get('quality', [])
    order_total_rule = any('order_total' in r.get('expression', '') and '>= 0' in r.get('expression', '')
                          for r in quality)
    status_rule = any('status_code' in r.get('expression', '') or 'status' in r.get('expression', '')
                     for r in quality)
    
    checks.append(("Orders: Non-negative order_total rule", order_total_rule))
    checks.append(("Orders: Status code validation rule", status_rule))
    
    return checks

def validate_scenario_3_thermostat(contract):
    """Validate Scenario 3: IoT requirements"""
    checks = []
    
    schema = contract.get('schema', {})
    properties = schema.get('properties', {})
    
    # Check temperature range in schema
    temp = properties.get('temperature_c', {})
    temp_range_ok = temp.get('minimum') == -30 and temp.get('maximum') == 60
    checks.append(("Thermostat: Temperature range [-30, 60] in schema", temp_range_ok))
    
    # Check battery range in schema
    battery = properties.get('battery_level', {})
    battery_range_ok = battery.get('minimum') == 0.0 and battery.get('maximum') == 1.0
    checks.append(("Thermostat: Battery range [0.0, 1.0] in schema", battery_range_ok))
    
    # Check quality rules
    quality = contract.get('quality', [])
    temp_rule = any('temperature_c' in r.get('expression', '') and 
                   '>= -30' in r.get('expression', '') and 
                   '<= 60' in r.get('expression', '')
                   for r in quality)
    battery_rule = any('battery_level' in r.get('expression', '')
                      for r in quality)
    
    checks.append(("Thermostat: Temperature range quality rule", temp_rule))
    checks.append(("Thermostat: Battery level quality rule", battery_rule))
    
    return checks

def validate_scenario_4_fintech(contract):
    """Validate Scenario 4: FinTech requirements"""
    checks = []
    
    schema = contract.get('schema', {})
    properties = schema.get('properties', {})
    
    # Check regex pattern in schema
    has_regex = any('pattern' in props and '^[A-Z0-9]{10}$' in props.get('pattern', '')
                   for field_name, props in properties.items()
                   if 'account' in field_name.lower())
    checks.append(("FinTech: Regex pattern ^[A-Z0-9]{10}$ found", has_regex))
    
    # Check hard enforcement
    quality = contract.get('quality', [])
    has_hard_enforcement = any(r.get('enforcement') == 'hard' 
                              for r in quality)
    checks.append(("FinTech: Hard circuit breaker enforcement", has_hard_enforcement))
    
    # Check circuit breaker documentation
    has_circuit_breaker_doc = any('BLOCK' in r.get('description', '').upper() or
                                  'circuit' in r.get('description', '').lower()
                                  for r in quality)
    checks.append(("FinTech: Hard circuit breaker documented", has_circuit_breaker_doc))
    
    return checks

def main():
    print("\n" + "="*60)
    print("DATA CONTRACT VALIDATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    contracts_dir = Path('datacontracts')
    
    scenarios = [
        ('rides_contract.yaml', 'Scenario 1: Ride-Share', validate_scenario_1_rides),
        ('orders_contract.yaml', 'Scenario 2: E-commerce Orders', validate_scenario_2_orders),
        ('thermostat_contract.yaml', 'Scenario 3: IoT Thermostat', validate_scenario_3_thermostat),
        ('fintech_contract.yaml', 'Scenario 4: FinTech Transactions', validate_scenario_4_fintech),
    ]
    
    all_passed = True
    total_checks = 0
    passed_checks = 0
    
    for filename, scenario_name, validator in scenarios:
        file_path = contracts_dir / filename
        
        print(f"--- Validating {scenario_name} ---")
        
        # YAML syntax check
        valid, error = validate_yaml_syntax(file_path)
        if not valid:
            print(f"[FAIL] YAML syntax error: {error}")
            all_passed = False
            continue
        
        # Load contract
        with open(file_path, 'r') as f:
            contract = yaml.safe_load(f)
        
        # ODCS structure check
        valid, missing = validate_odcs_structure(contract)
        if not valid:
            print(f"[FAIL] Missing ODCS sections: {missing}")
            all_passed = False
            continue
        
        # Scenario-specific checks
        checks = validator(contract)
        for check_name, passed in checks:
            total_checks += 1
            status = "PASS" if passed else "FAIL"
            symbol = "✓" if passed else "✗"
            print(f"[{status}] {check_name} {symbol}")
            if passed:
                passed_checks += 1
            else:
                all_passed = False
        
        print()
    
    print("="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print()
    
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("All contracts meet assignment requirements!")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("Please review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
